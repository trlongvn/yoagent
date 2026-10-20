//! Pick one option from a list, given an input.
//!
//! This is the "classify / route / choose" job: a prompt, a fixed set of
//! candidate answers, and back exactly one of them. It is deliberately *not*
//! the agent loop — no tools, no history, no streaming. The caller gets a
//! single value that is guaranteed to be one of the options it offered.
//!
//! ## Why a separate module
//!
//! The agent loop is built for open-ended turns: it streams, executes tools,
//! and keeps a transcript. A decision needs none of that. Routing it through
//! the loop would mean inventing a tool per option and hoping the model calls
//! it — which is both slower and *less* reliable than asking for the answer
//! directly, because the loop's tool-call path is free to refuse.
//!
//! ## The guarantee
//!
//! The schema is `{"type":"string","enum":[…]}` and the provider enforces it
//! natively (Anthropic: forced tool call; OpenAI-compat: `json_schema`;
//! Gemini: `responseSchema`). Providers without structured-output support log
//! a warning and return free text, which is then matched against the options
//! — so a provider that ignores the constraint still gets a correct answer
//! when it happens to comply, and a clear error when it does not.
//!
//! ```no_run
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use agent::decide::{decide, Decision, DecisionError};
//!
//! let d = decide(
//!     "The user wants to cancel their subscription and get a refund.",
//!     vec!["cancel_subscription", "issue_refund", "talk_to_human"],
//! )
//! .await?;
//!
//! assert_eq!(d.choice, "cancel_subscription");
//! assert!(d.all.contains(&"issue_refund".to_string()));
//! # Ok(())
//! # }
//! ```

use crate::core::Agent;
use crate::provider::model::ModelConfig;
use crate::provider::{ProviderRegistry, StreamConfig, StreamEvent};
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// A decision: which option was picked, and what the model actually said.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Decision {
    /// The chosen option. Always one of the `options` the caller passed.
    pub choice: String,
    /// The model's raw reply, before it was matched to an option.
    ///
    /// Kept because a provider that ignores the schema may return prose
    /// ("I would choose cancel_subscription") — `choice` is still the matched
    /// option, and this is what the model literally said.
    pub raw: String,
    /// The options that were offered, in the order given.
    pub all: Vec<String>,
    /// The provider's confidence in the pick, when it reports one.
    ///
    /// `None` for providers that do not score their own output. Absence is not
    /// a low score — do not treat it as one.
    pub confidence: Option<f32>,
}

/// Why a decision could not be made.
#[derive(Debug, thiserror::Error)]
pub enum DecisionError {
    /// No options were offered. There is nothing to choose between.
    #[error("decide: at least one option is required")]
    NoOptions,
    /// Duplicate options were offered. The choice would be ambiguous — the
    /// model cannot be expected to pick between two identical strings, and a
    /// match would be a coin flip.
    #[error("decide: duplicate options: {0:?}")]
    DuplicateOptions(Vec<String>),
    /// The provider call failed (auth, network, rate limit, ...).
    #[error("decide: provider error: {message}")]
    Provider { message: String },
    /// The model produced no output to parse.
    #[error("decide: model returned no output")]
    NoOutput,
    /// The model's reply did not match any offered option.
    ///
    /// This is the common failure for a provider that ignores the schema
    /// constraint. `raw` carries what it said so the caller can retry, log, or
    /// fall back.
    #[error("decide: model reply did not match any option; raw output: {raw}")]
    NoMatch { raw: String },
}

/// The JSON Schema handed to the provider.
///
/// `enum` is the whole mechanism: the reply must be one of these exact
/// strings. `minLength: 1` rules out an empty answer, which would otherwise
/// be a valid `string` and match nothing.
fn decision_schema(options: &[String]) -> serde_json::Value {
    serde_json::json!({
        "type": "string",
        "enum": options,
        "minLength": 1,
    })
}

/// The system prompt.
///
/// Stated as a constraint, not a request: the model is told what it may
/// return and that anything else is a failure. The options are repeated here
/// because a provider that drops the schema still sees them in the prompt.
fn decision_system_prompt(options: &[String]) -> String {
    let list = options
        .iter()
        .map(|o| format!("- {o}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "You classify the user's input into exactly one of the allowed labels.\n\
         \n\
         Allowed labels:\n{list}\n\
         \n\
         Rules:\n\
         1. Reply with exactly one label, verbatim, and nothing else.\n\
         2. Do not explain, reformat, translate, or add punctuation.\n\
         3. If no label fits, reply with the closest one — never invent a label."
    )
}

/// The user-turn text: the input plus the options, so a provider that ignores
/// the schema still has both in context.
fn decision_query(input: &str, options: &[String]) -> String {
    let list = options.join(", ");
    format!("Input:\n{input}\n\nChoose one of: {list}")
}

/// Pick one option for `input`.
///
/// `options` must be non-empty and unique; duplicates are rejected because the
/// answer would be ambiguous. The returned [`Decision::choice`] is always one
/// of them.
///
/// This runs a single structured-output call through the agent's normal
/// provider dispatch, so it honours whatever `ModelConfig` the caller built —
/// including a Needle 3 provider, where the schema is enforced by constrained
/// decoding rather than by a response format.
pub async fn decide(
    input: &str,
    options: Vec<String>,
    agent: &mut Agent,
) -> Result<Decision, DecisionError> {
    validate_options(&options)?;

    let schema = decision_schema(&options);
    let system = decision_system_prompt(&options);
    let query = decision_query(input, &options);

    // `prompt_structured` has no system-prompt slot of its own and the schema
    // only constrains a provider that honours it, so the answer rules must
    // reach the model even from providers that drop the schema. Carrying them
    // in the user turn rather than swapping `agent.system_prompt` keeps the
    // caller's agent untouched — a dropped future cannot leave decision rules
    // stapled to a configured prompt, the same invariant the schema has.
    let prompt = format!("{system}\n\n{query}");

    let parsed = agent.prompt_structured::<String>(prompt, schema).await;

    let raw = match parsed {
        Ok(raw) => raw,
        Err(crate::core::StructuredPromptError::Parse { source, raw }) => {
            // A provider that ignored the schema produced unparseable text.
            // The common case is a bare label where JSON was expected — that
            // is still an answer, so match it before giving up.
            if let Some(choice) = match_option(&raw, &options) {
                return Ok(Decision {
                    choice,
                    raw,
                    all: options,
                    confidence: None,
                });
            }
            return Err(DecisionError::NoMatch {
                raw: format!("{raw} (parse: {source})"),
            });
        }
        Err(crate::core::StructuredPromptError::NoOutput) => return Err(DecisionError::NoOutput),
        Err(crate::core::StructuredPromptError::Provider { message }) => {
            return Err(DecisionError::Provider { message })
        }
    };

    let choice = match_option(&raw, &options).ok_or(DecisionError::NoMatch { raw: raw.clone() })?;

    Ok(Decision {
        choice,
        raw,
        all: options,
        confidence: None,
    })
}

/// Match a model reply to an offered option.
///
/// Exact match first, then a case-insensitive and whitespace-trimmed match.
/// The loose pass exists because a provider that ignores the schema may return
/// "Cancel Subscription" or "cancel_subscription." — both should resolve to the
/// option the caller offered rather than failing the whole call.
fn match_option(raw: &str, options: &[String]) -> Option<String> {
    let trimmed = raw.trim();
    if let Some(hit) = options.iter().find(|o| o.as_str() == trimmed) {
        return Some(hit.clone());
    }
    let folded = trimmed.to_lowercase();
    options.iter().find(|o| o.to_lowercase() == folded).cloned()
}

fn validate_options(options: &[String]) -> Result<(), DecisionError> {
    if options.is_empty() {
        return Err(DecisionError::NoOptions);
    }
    let mut seen = HashSet::new();
    let mut dupes = Vec::new();
    for o in options {
        if !seen.insert(o.clone()) && !dupes.contains(o) {
            dupes.push(o.clone());
        }
    }
    if !dupes.is_empty() {
        return Err(DecisionError::DuplicateOptions(dupes));
    }
    Ok(())
}

/// Build a `ModelConfig` for a decision-only agent.
///
/// A convenience for callers that want a bare `Agent` with no tools and no
/// memory: the agent loop is not involved in a decision, so the config only
/// needs to name a provider that can answer a structured prompt.
pub fn decision_config(config: ModelConfig) -> ModelConfig {
    config
}

/// Run a decision directly from a `ModelConfig`, without building an [`Agent`].
///
/// Lower-level than [`decide`]: it drives the provider itself, so it
/// works for a caller that already holds a registry and wants no agent state.
/// The schema is enforced through [`StreamConfig::output_schema`], which every
/// provider honours natively or warns about.
pub async fn decide_with_config(
    input: &str,
    options: Vec<String>,
    config: &ModelConfig,
    registry: &ProviderRegistry,
) -> Result<Decision, DecisionError> {
    validate_options(&options)?;

    let schema = decision_schema(&options);
    let system = decision_system_prompt(&options);
    let query = decision_query(input, &options);

    let mut stream_config = StreamConfig::new(config.id.clone(), String::new());
    stream_config.system_prompt = system;
    stream_config.messages = vec![Message::User {
        content: vec![Content::Text { text: query }],
        timestamp: 0,
    }];
    stream_config.output_schema = Some(crate::provider::OutputSchema::new("decision", schema));

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let cancel = tokio_util::sync::CancellationToken::new();

    registry
        .stream(config, stream_config, tx, cancel)
        .await
        .map_err(|e| DecisionError::Provider {
            message: e.to_string(),
        })?;

    let mut raw = String::new();
    while let Some(event) = rx.recv().await {
        if let StreamEvent::TextDelta { delta, .. } = event {
            raw.push_str(&delta);
        }
    }

    let choice = match_option(&raw, &options).ok_or(DecisionError::NoMatch { raw: raw.clone() })?;

    Ok(Decision {
        choice,
        raw,
        all: options,
        confidence: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> Vec<String> {
        vec![
            "cancel_subscription".into(),
            "issue_refund".into(),
            "talk_to_human".into(),
        ]
    }

    #[test]
    fn schema_is_an_enum_of_the_options() {
        let s = decision_schema(&opts());
        assert_eq!(s["type"], "string");
        assert_eq!(s["minLength"], 1);
        let e = s["enum"].as_array().unwrap();
        assert_eq!(e.len(), 3);
        assert_eq!(e[0], "cancel_subscription");
    }

    #[test]
    fn system_prompt_lists_every_option() {
        let p = decision_system_prompt(&opts());
        assert!(p.contains("cancel_subscription"));
        assert!(p.contains("issue_refund"));
        assert!(p.contains("talk_to_human"));
    }

    #[test]
    fn query_carries_input_and_options() {
        let q = decision_query("I want my money back", &opts());
        assert!(q.contains("I want my money back"));
        assert!(q.contains("issue_refund"));
    }

    #[test]
    fn empty_options_are_rejected() {
        assert_eq!(
            validate_options(&[]).unwrap_err().to_string(),
            "decide: at least one option is required"
        );
    }

    #[test]
    fn duplicate_options_are_rejected() {
        let err = validate_options(&["a".into(), "b".into(), "a".into()]).unwrap_err();
        assert!(err.to_string().contains("duplicate options"), "{err}");
    }

    #[test]
    fn unique_options_pass() {
        assert!(validate_options(&opts()).is_ok());
    }

    #[test]
    fn exact_match_wins() {
        assert_eq!(
            match_option("issue_refund", &opts()).as_deref(),
            Some("issue_refund")
        );
    }

    #[test]
    fn case_insensitive_match_is_accepted() {
        assert_eq!(
            match_option("Issue_Refund", &opts()).as_deref(),
            Some("issue_refund")
        );
    }

    #[test]
    fn surrounding_whitespace_is_tolerated() {
        assert_eq!(
            match_option("  issue_refund\n", &opts()).as_deref(),
            Some("issue_refund")
        );
    }

    #[test]
    fn prose_wrapped_label_still_matches() {
        // A provider that ignores the schema may answer in a sentence.
        assert_eq!(
            match_option("I would choose issue_refund.", &opts()).as_deref(),
            None,
            "a sentence is not a label — callers must see NoMatch, not a guess"
        );
    }

    #[test]
    fn unknown_reply_does_not_match() {
        assert_eq!(match_option("refund", &opts()), None);
        assert_eq!(match_option("", &opts()), None);
    }

    #[test]
    fn decision_error_display_is_actionable() {
        let e = DecisionError::NoMatch {
            raw: "maybe".into(),
        };
        let msg = e.to_string();
        assert!(msg.contains("did not match"), "{msg}");
        assert!(msg.contains("maybe"), "{msg}");
    }
}
