//! Needle 3 on-device tool-calling provider.
//!
//! Needle 3 is a 2-bit, 8–29 MB generator trained for one job: given a query
//! and a tool list, emit a JSON tool call. It is *not* a general-purpose chat
//! model — it emits either a `<tool_call>` payload or `[]` (a refusal), and
//! has no path to free-form prose. This provider is therefore a **front-door
//! tool-caller**, not a complete LLM replacement:
//!
//! - High-confidence call → handed straight to the agent loop's tool executor.
//! - Low confidence, refusal, or any failure → [`ProviderError::Other`] with a
//!   stable `needle3:` prefix, so a routing layer can escalate to a larger
//!   model without pattern-matching on prose.
//!
//! ## Why constrained decoding is the point
//!
//! The engine compiles a byte-level grammar from the tool schemas and masks
//! logits between the `<tool_call>` markers. The payload *cannot* name an
//! undeclared tool or a nonexistent argument — the guarantee a normal LLM
//! provider gets only by trusting the model's JSON. `constrain` is scoped to
//! the markers deliberately: a `"name":` inside `<think>` must not drag the
//! state machine into a constrained state where it does not belong.
//!
//! ## Threading
//!
//! `V3Engine` is `Send + Sync` (asserted upstream), but generation is
//! synchronous and CPU-bound, so it runs on [`tokio::task::spawn_blocking`]
//! rather than blocking an async worker. **The engine is stateless** — every
//! call allocates a fresh `V3Cache` and prefill — so nothing is carried
//! between turns and the whole model is released while idle.

use super::model::ApiProtocol;
use super::traits::*;
use crate::types::*;
use async_trait::async_trait;
use needle_infer::v3_engine::{
    extract_tool_call, StopReason as NeedleStop, V3Engine, V3EngineError, V3Options, V3Result,
};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

/// Default confidence floor for trusting a Needle 3 call.
///
/// The upstream checkpoint scores ~0.93 on a correct completion and ~0.26 on
/// a wrong one, so the default sits far above the failure band. Callers
/// should still calibrate against their own tool set — a tool with many
/// ambiguous arguments legitimately scores lower.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Default path to the Needle 3 weights, resolved at construction time.
pub const DEFAULT_WEIGHTS_PATH: &str = "models/needle3.cact";

/// Why this provider did not produce a call.
///
/// Each variant carries a stable `needle3:` prefix so a router can match on
/// *intent* rather than on free-form prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Needle3Outcome {
    /// The model returned `[]` — a deliberate refusal, not a failure.
    Refused,
    /// Confidence was below the configured floor; escalate.
    LowConfidence,
    /// The container carries no confidence head, so the call is unscored.
    /// Treat as untrusted.
    Unscored,
    /// The engine produced no `<tool_call>` markers at all.
    NoCall,
}

impl std::fmt::Display for Needle3Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused => write!(f, "needle3:refused"),
            Self::LowConfidence => write!(f, "needle3:low_confidence"),
            Self::Unscored => write!(f, "needle3:unscored"),
            Self::NoCall => write!(f, "needle3:no_call"),
        }
    }
}

/// Why a Needle 3 provider could not be constructed.
///
/// Distinct from [`ProviderError`], which describes a failure to *answer* a
/// request. This one means "the model is not usable here at all", and it
/// happens before the first turn rather than during one.
#[derive(Debug, thiserror::Error)]
pub enum Needle3BuildError {
    /// The `.cact` container could not be read or parsed. Check the path and
    /// that the file is a Needle 3 archive (the generation is read from the
    /// container, not the file name).
    #[error("needle3: cannot load weights: {0}")]
    Engine(#[source] V3EngineError),
}

/// A Needle 3 model bound to a fixed tool set.
pub struct Needle3Provider {
    /// Behind an `Arc` so [`spawn_blocking`](tokio::task::spawn_blocking) can
    /// capture an owned handle — `stream(&self)` cannot move a borrow into a
    /// `'static` closure, and generation is synchronous CPU work.
    engine: std::sync::Arc<V3Engine>,
    /// Tool schemas as Needle expects them, serialized once at construction.
    /// Needle's grammar is compiled from this string, so it is built here
    /// rather than per-turn.
    tools_json: String,
    confidence_threshold: f32,
    /// Refusals are a legitimate answer ("that is not something these tools do"),
    /// so they do not escalate by default.
    escalate_on_refusal: bool,
    /// Emit `<think>` as `Content::Thinking` blocks. Off by default: Needle's
    /// reasoning is terse and task-local, and surfacing it to the user is a
    /// product decision rather than a safe default.
    surface_reasoning: bool,
    /// KV cache precision. `Int8` matches the container's `kv_bits` and costs
    /// about a quarter of the memory; not bit-identical to the f32 path, so
    /// it stays opt-in.
    kv_int8: bool,
    /// Generation cap. Needle 3 defaults to 256 because it reasons before
    /// answering; this overrides it.
    max_new_tokens: usize,
}

impl std::fmt::Debug for Needle3Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Needle3Provider")
            .field("tools", &count_tools(&self.tools_json))
            .field("confidence_threshold", &self.confidence_threshold)
            .field("escalate_on_refusal", &self.escalate_on_refusal)
            .field("surface_reasoning", &self.surface_reasoning)
            .field("kv_int8", &self.kv_int8)
            .field("max_new_tokens", &self.max_new_tokens)
            .field("has_confidence_head", &self.engine.confidence.is_some())
            .finish_non_exhaustive()
    }
}

fn count_tools(tools_json: &str) -> usize {
    serde_json::from_str::<Value>(tools_json)
        .ok()
        .and_then(|v| v.as_array().map(|a| a.len()))
        .unwrap_or(0)
}

/// Resolve the weights path from an explicit value or the conventional env var.
pub fn resolve_weights_path(explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Some(p) = std::env::var_os("NEEDLE3_WEIGHTS") {
        return PathBuf::from(p);
    }
    PathBuf::from(DEFAULT_WEIGHTS_PATH)
}

impl Needle3Provider {
    /// Load a `.cact` file and bind a tool set.
    ///
    /// `tools` are the same [`ToolDefinition`] values every provider receives;
    /// they are converted to Needle's schema shape and serialized once.
    pub fn new(weights: Option<&Path>, tools: Vec<ToolDefinition>) -> Result<Self, V3EngineError> {
        Self::with_threshold(weights, tools, DEFAULT_CONFIDENCE_THRESHOLD)
    }

    /// Like [`new`](Self::new) with an explicit confidence floor.
    pub fn with_threshold(
        weights: Option<&Path>,
        tools: Vec<ToolDefinition>,
        confidence_threshold: f32,
    ) -> Result<Self, V3EngineError> {
        let path = resolve_weights_path(weights);
        let engine = V3Engine::load(&path)?;
        Ok(Self {
            tools_json: tools_to_needle_json(&tools),
            engine: std::sync::Arc::new(engine),
            confidence_threshold,
            escalate_on_refusal: false,
            surface_reasoning: false,
            kv_int8: false,
            max_new_tokens: 256,
        })
    }

    /// Weight a refusal as an escalation trigger.
    ///
    /// Off by default because a refusal is often the *right* answer — the model
    /// is scoped to its tools and "none of these do that" is a real signal.
    pub fn escalate_on_refusals(mut self) -> Self {
        self.escalate_on_refusal = true;
        self
    }

    /// Surface the model's `<think>` trace as a `Content::Thinking` block.
    pub fn with_reasoning(mut self) -> Self {
        self.surface_reasoning = true;
        self
    }

    /// Use the 8-bit KV cache (quarter the memory).
    pub fn with_kv_int8(mut self) -> Self {
        self.kv_int8 = true;
        self
    }

    /// Override the generation token budget.
    pub fn with_max_new_tokens(mut self, max_new_tokens: usize) -> Self {
        self.max_new_tokens = max_new_tokens;
        self
    }

    /// The model's reasoning for a completion, if it emitted any.
    pub fn reasoning_of<'a>(&self, text: &'a str) -> Option<&'a str> {
        V3Engine::reasoning(text)
    }

    /// Run the model without the async plumbing — used by tests and by
    /// callers that want the raw payload.
    pub fn complete(&self, query: &str, system: Option<&str>) -> Completion {
        generate(
            &self.engine,
            &self.tools_json,
            query,
            system,
            self.max_new_tokens,
            self.kv_int8,
        )
    }
}

/// Run one Needle 3 completion.
///
/// A free function so both [`Needle3Provider::complete`] and the
/// `spawn_blocking` closure in [`stream`](StreamProvider::stream) can call it
/// without moving a borrow into a `'static` closure.
fn generate(
    engine: &V3Engine,
    tools_json: &str,
    query: &str,
    system: Option<&str>,
    max_new_tokens: usize,
    kv_int8: bool,
) -> Completion {
    let opts = V3Options {
        max_new_tokens,
        temperature: 0.0,
        seed: 0,
        system: system.map(str::to_owned),
        constrain: true,
        kv_precision: if kv_int8 {
            needle_infer::v3_engine::KvPrecision::Int8
        } else {
            needle_infer::v3_engine::KvPrecision::F32
        },
    };
    let V3Result {
        text, tokens, stop, ..
    } = engine.generate(query, tools_json, &opts);

    // Score the *completion*, never the bare query — the head is trained on
    // finished answers, and a bare query silently scores ~0.80.
    let confidence = engine.confidence_for(query, tools_json, &text);

    // A free function, not a `V3Engine` method — the marker scan is engine-
    // agnostic and shared with the parser.
    let payload = extract_tool_call(&text);
    let calls = match &payload {
        Some(raw) => parse_call(raw),
        None => Vec::new(),
    };

    Completion {
        text,
        tokens_generated: tokens.len(),
        stop: match stop {
            NeedleStop::Eos | NeedleStop::ImEnd => StopReason::Stop,
            NeedleStop::MaxTokens => StopReason::Length,
            NeedleStop::MaxSeqLen => StopReason::Length,
        },
        confidence,
        calls,
        refused: matches!(payload.as_deref(), Some("[]")),
    }
}

/// A raw Needle 3 completion, before it becomes a [`Message`].
#[derive(Debug, Clone)]
pub struct Completion {
    /// Full generated text: `<think>…</think>` followed by the call.
    pub text: String,
    pub tokens_generated: usize,
    pub stop: StopReason,
    pub confidence: Option<f32>,
    pub calls: Vec<Content>,
    /// The model emitted `[]` — a deliberate refusal.
    pub refused: bool,
}

impl Completion {
    /// Whether this completion should be trusted without escalation.
    pub fn trusted(&self, threshold: f32) -> Result<(), Needle3Outcome> {
        if self.calls.is_empty() {
            if self.refused {
                return Err(Needle3Outcome::Refused);
            }
            return Err(Needle3Outcome::NoCall);
        }
        match self.confidence {
            None => Err(Needle3Outcome::Unscored),
            Some(c) if c < threshold => Err(Needle3Outcome::LowConfidence),
            _ => Ok(()),
        }
    }
}

/// Convert this crate's tool definitions into the schema shape Needle's
/// grammar understands.
///
/// Needle accepts flat (`{"name","arguments"}`) and JSON-Schema
/// (`{"name","parameters":{"type":"object","properties":…}}`) shapes. The agent
/// loop's [`ToolDefinition`] is `{name, description, parameters}`, so it maps
/// straight across. `description` is dropped from the grammar-relevant payload
/// only because Needle's `ToolDef` does not carry it — it is preserved by the
/// engine's prompt builder from the full schema. We serialize the original
/// `parameters` object verbatim so every constraint (`enum`, `minimum`,
/// `pattern`, …) reaches the grammar.
fn tools_to_needle_json(tools: &[ToolDefinition]) -> String {
    let arr: Vec<Value> = tools
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        })
        .collect();
    serde_json::to_string(&arr).unwrap_or_else(|_| "[]".into())
}

/// Parse the `<tool_call>` payload into content blocks.
///
/// The payload is `[{"name":…,"arguments":{…}}]`. A parse failure is not fatal
/// for the engine's own grammar — an unterminated call still carries the prefix
/// it managed to emit — but it *is* fatal here, because a malformed call would
/// be executed by the agent loop. Returning empty and letting the caller see
/// `NoCall` is safer than guessing.
fn parse_call(raw: &str) -> Vec<Content> {
    let parsed: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let items = match parsed {
        // Needle's grammar always emits a `[{…}]` array. A bare object is
        // not a shape this engine can produce, so it is treated as malformed
        // rather than guessed at.
        Value::Array(a) => a,
        _ => return Vec::new(),
    };

    items
        .into_iter()
        .enumerate()
        .filter_map(|(i, item)| {
            let name = item.get("name")?.as_str()?.to_owned();
            let arguments = match item.get("arguments") {
                Some(Value::Object(m)) => Value::Object(m.clone()),
                Some(other) => other.clone(),
                None => Value::Object(Map::new()),
            };
            // Needle assigns no call ids; the agent loop's executor only needs
            // them unique within the turn to match tool results back.
            let id = format!("needle3-{}", i);
            Some(Content::tool_call(id, name, arguments))
        })
        .collect()
}

#[async_trait]
impl StreamProvider for Needle3Provider {
    async fn stream(
        &self,
        config: StreamConfig,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<Message, ProviderError> {
        let _ = tx.send(StreamEvent::Start);

        // Needle 3 is a pure tool-call generator: it consumes the latest user
        // turn plus a fixed tool list, and there is no transcript to replay.
        // Earlier user messages are not sent to the model, but they are not
        // dropped from the agent's history — the loop keeps them.
        let query = last_user_text(&config.messages)
            .ok_or_else(|| ProviderError::Other("needle3:no_user_turn".into()))?;

        let system = config.system_prompt.clone();
        let query = query.clone();
        // Clone the `Arc` rather than moving a `&self` in: `spawn_blocking`
        // requires `'static`, and an owned handle keeps the engine alive for
        // the whole generation without borrowing the caller's frame.
        let engine = self.engine.clone();
        let tools_json = self.tools_json.clone();
        let threshold = self.confidence_threshold;
        let surface_reasoning = self.surface_reasoning;
        let max_new_tokens = self.max_new_tokens;
        let kv_int8 = self.kv_int8;

        let completion = tokio::task::spawn_blocking(move || {
            generate(
                &engine,
                &tools_json,
                &query,
                if system.is_empty() {
                    None
                } else {
                    Some(&system)
                },
                max_new_tokens,
                kv_int8,
            )
        })
        .await
        .map_err(|e| ProviderError::Other(format!("needle3:join_error:{e}")))?;

        if cancel.is_cancelled() {
            return Err(ProviderError::Cancelled);
        }

        // Stream the reasoning trace as deltas, then the call. Needle's
        // `generate_with` already hands decoded text (not raw SentencePiece
        // pieces), so concatenation is exact.
        let (thinking, call_text) = split_think(&completion.text);
        if surface_reasoning && !thinking.is_empty() {
            let _ = tx.send(StreamEvent::ThinkingDelta {
                content_index: 0,
                delta: thinking.to_owned(),
            });
        }
        if !call_text.is_empty() {
            let _ = tx.send(StreamEvent::TextDelta {
                content_index: 0,
                delta: call_text.to_owned(),
            });
        }
        for (i, call) in completion.calls.iter().enumerate() {
            if let Content::ToolCall { id, name, .. } = call {
                let _ = tx.send(StreamEvent::ToolCallStart {
                    content_index: i,
                    id: id.clone(),
                    name: name.clone(),
                });
                let _ = tx.send(StreamEvent::ToolCallDelta {
                    content_index: i,
                    delta: String::new(),
                });
                let _ = tx.send(StreamEvent::ToolCallEnd { content_index: i });
            }
        }

        // Gate before building the message: an untrusted call must never reach
        // the tool executor.
        if let Err(outcome) = completion.trusted(threshold) {
            let detail = match completion.confidence {
                Some(c) => format!("{outcome}:{c:.3}"),
                None => outcome.to_string(),
            };
            return Err(ProviderError::Other(detail));
        }

        let mut content: Vec<Content> = Vec::new();
        if surface_reasoning && !thinking.is_empty() {
            content.push(Content::thinking(thinking.to_owned()));
        }
        content.extend(completion.calls.iter().cloned());

        // No tool call means a refusal — that is a legitimate terminal answer,
        // not an error.
        let stop_reason = if content_has_tool_call(&content) {
            StopReason::ToolUse
        } else {
            StopReason::Stop
        };

        let message = Message::Assistant {
            content,
            stop_reason,
            model: "needle3".into(),
            provider: "needle3".into(),
            // Needle 3 reports no token accounting; the output count is the
            // only figure it can supply. Zeroed elsewhere so ContextTracker
            // does not read a missing figure as a tiny one.
            usage: Usage {
                input: 0,
                output: completion.tokens_generated as u64,
                cache_read: 0,
                cache_write: 0,
                total_tokens: completion.tokens_generated as u64,
            },
            timestamp: now_ms(),
            error_message: None,
        };

        let _ = tx.send(StreamEvent::Done {
            message: message.clone(),
        });
        Ok(message)
    }

    fn protocol(&self) -> Option<ApiProtocol> {
        Some(ApiProtocol::Needle3)
    }
}

fn content_has_tool_call(content: &[Content]) -> bool {
    content
        .iter()
        .any(|c| matches!(c, Content::ToolCall { .. }))
}

/// The plain text of the most recent user turn.
///
/// Needle 3 takes a single `(query, tools)` pair, not a transcript. When the
/// last user turn carries several text blocks they are joined; tool-result
/// blocks are skipped because a tool result is the *previous* turn's output,
/// which Needle 3 already consumed when it produced the call.
fn last_user_text(messages: &[Message]) -> Option<String> {
    let text = messages.iter().rev().find_map(|m| match m {
        Message::User { content, .. } => Some(content),
        _ => None,
    })?;
    let mut parts = text.iter().filter_map(|c| match c {
        Content::Text { text } => Some(text.clone()),
        _ => None,
    });
    let first = parts.next()?;
    let joined: Vec<String> = std::iter::once(first).chain(parts).collect();
    let out = joined.join("\n");
    if out.trim().is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Split a completion into its reasoning trace and the remainder.
///
/// Needle 3 emits `<think>…</think>` before the call; an unterminated trace
/// still contains reasoning, and the remainder is then empty.
fn split_think(text: &str) -> (&str, &str) {
    match (text.find("<think>"), text.find("</think>")) {
        (Some(start), Some(end)) => {
            let after = end + "</think>".len();
            (&text[start + "<think>".len()..end], text[after..].trim())
        }
        (Some(start), None) => (&text[start + "<think>".len()..], ""),
        (None, _) => ("", text.trim()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tools() -> Vec<ToolDefinition> {
        serde_json::from_str(
            r#"[{
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }]"#,
        )
        .expect("valid tool json")
    }

    #[test]
    fn tools_serialize_in_needle_schema_shape() {
        let out = tools_to_needle_json(&weather_tools());
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 1);
        assert_eq!(v[0]["name"], "get_weather");
        // The parameters object survives verbatim — the grammar needs `enum`.
        assert_eq!(
            v[0]["parameters"]["properties"]["unit"]["enum"][0],
            "celsius"
        );
    }

    #[test]
    fn parse_call_maps_schema_and_arguments() {
        let calls =
            parse_call(r#"[{"name":"get_weather","arguments":{"city":"Paris","unit":"celsius"}}]"#);
        match &calls[..] {
            [Content::ToolCall {
                id,
                name,
                arguments,
                ..
            }] => {
                assert_eq!(name, "get_weather");
                assert!(id.starts_with("needle3-"));
                assert_eq!(arguments["city"], "Paris");
                assert_eq!(arguments["unit"], "celsius");
            }
            other => panic!("expected one tool call, got {other:?}"),
        }
    }

    #[test]
    fn parse_call_handles_empty_arguments_and_absent_key() {
        let calls = parse_call(r#"[{"name":"ping"}]"#);
        match &calls[..] {
            [Content::ToolCall {
                name, arguments, ..
            }] => {
                assert_eq!(name, "ping");
                assert!(arguments.as_object().unwrap().is_empty());
            }
            other => panic!("expected one tool call, got {other:?}"),
        }
    }

    #[test]
    fn parse_call_rejects_malformed_payload() {
        assert!(parse_call(r#"[{"name":"get_w"#).is_empty());
        assert!(parse_call("not json").is_empty());
        assert!(parse_call(r#"{"name":"x"}"#).is_empty());
    }

    #[test]
    fn parse_call_rejects_a_bare_object() {
        // Needle's grammar emits `[{…}]`, so a bare object is not a shape the
        // engine can produce — it must be rejected, not guessed at.
        assert_eq!(
            parse_call(r#"{"name":"get_weather","arguments":{}}"#).len(),
            0
        );
    }

    #[test]
    fn refusal_payload_is_not_a_call() {
        let calls = parse_call("[]");
        assert!(calls.is_empty());
    }

    #[test]
    fn split_think_separates_trace_from_call() {
        let (think, rest) = split_think(
            "<think>cities are strings</think>[{\"name\":\"get_weather\",\"arguments\":{}}]",
        );
        assert_eq!(think, "cities are strings");
        assert!(rest.contains("get_weather"));
    }

    #[test]
    fn split_think_tolerates_unterminated_trace() {
        let (think, rest) = split_think("<think>still reasoning");
        assert_eq!(think, "still reasoning");
        assert_eq!(rest, "");
    }

    #[test]
    fn split_think_without_trace_returns_whole_text() {
        let (think, rest) = split_think("[{\"name\":\"ping\"}]");
        assert_eq!(think, "");
        assert!(rest.contains("ping"));
    }

    #[test]
    fn last_user_text_reads_the_latest_user_turn() {
        let msgs = vec![
            Message::User {
                content: vec![Content::Text {
                    text: "first".into(),
                }],
                timestamp: 1,
            },
            Message::Assistant {
                content: vec![Content::ToolCall {
                    id: "needle3-0".into(),
                    name: "get_weather".into(),
                    arguments: json!({"city": "Paris"}),
                    provider_metadata: None,
                }],
                stop_reason: StopReason::ToolUse,
                model: "needle3".into(),
                provider: "needle3".into(),
                usage: Usage::default(),
                timestamp: 2,
                error_message: None,
            },
            Message::User {
                content: vec![Content::Text {
                    text: "and now in Tokyo".into(),
                }],
                timestamp: 3,
            },
        ];
        assert_eq!(last_user_text(&msgs).as_deref(), Some("and now in Tokyo"));
    }

    #[test]
    fn last_user_text_joins_multiple_text_blocks() {
        let msgs = vec![Message::User {
            content: vec![
                Content::Text {
                    text: "line one".into(),
                },
                Content::Text {
                    text: "line two".into(),
                },
            ],
            timestamp: 1,
        }];
        assert_eq!(last_user_text(&msgs).as_deref(), Some("line one\nline two"));
    }

    #[test]
    fn last_user_text_skips_non_text_content() {
        let msgs = vec![Message::User {
            content: vec![Content::Image {
                data: "AAAA".into(),
                mime_type: "image/png".into(),
            }],
            timestamp: 1,
        }];
        assert_eq!(last_user_text(&msgs), None);
    }

    #[test]
    fn last_user_text_none_without_user_turn() {
        assert_eq!(last_user_text(&[]), None);
    }

    #[test]
    fn completion_trust_gate_distinguishes_outcomes() {
        let trusted = Completion {
            text: "x".into(),
            tokens_generated: 1,
            stop: StopReason::ToolUse,
            confidence: Some(0.93),
            calls: vec![Content::tool_call("needle3-0", "get_weather", json!({}))],
            refused: false,
        };
        assert_eq!(trusted.trusted(0.5), Ok(()));

        let mut low = trusted.clone();
        low.confidence = Some(0.26);
        assert_eq!(low.trusted(0.5), Err(Needle3Outcome::LowConfidence));

        let mut unscored = trusted.clone();
        unscored.confidence = None;
        assert_eq!(unscored.trusted(0.5), Err(Needle3Outcome::Unscored));

        let mut refused = trusted.clone();
        refused.calls = vec![];
        refused.refused = true;
        assert_eq!(refused.trusted(0.5), Err(Needle3Outcome::Refused));

        let mut nocall = trusted.clone();
        nocall.calls = vec![];
        nocall.refused = false;
        assert_eq!(nocall.trusted(0.5), Err(Needle3Outcome::NoCall));
    }

    #[test]
    fn outcome_strings_carry_stable_prefix() {
        assert_eq!(Needle3Outcome::Refused.to_string(), "needle3:refused");
        assert_eq!(
            Needle3Outcome::LowConfidence.to_string(),
            "needle3:low_confidence"
        );
        assert_eq!(Needle3Outcome::NoCall.to_string(), "needle3:no_call");
    }

    #[test]
    fn provider_reports_its_protocol() {
        // Constructing the engine is expensive and needs real weights; assert
        // the wiring without touching disk.
        assert_eq!(ApiProtocol::Needle3.to_string(), "needle3");
    }
}
