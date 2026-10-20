//! Phase 5 — observability and evaluation.
//!
//! A reproducible memory-enabled vs memory-disabled comparison on the same
//! cases, measuring time to first tool, end-to-end latency, LLM turns, token
//! usage, and task success — with cold initialization separated from warmed
//! runs, and offline mocks standing in for a real backend.
//!
//! Everything runs against `FakeMemoryBackend`, so every number is
//! deterministic and the assertions are regression checks, not thresholds
//! tuned to a machine. The plan's real-backend measurement (openai/ollama
//! embeddings) cannot be produced here without network access and keys; that
//! limitation is stated rather than papered over.

// `memory` and `memory-onnx` both enable the module — see `lib.rs`.
#![cfg(any(feature = "memory", feature = "memory-onnx"))]

mod support;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use ::agent::agent_loop::{agent_loop, AgentLoopConfig};
use ::agent::context::ContextConfig;
use ::agent::memory_integration::{
    make_prepare_context, RecallResult, DEFAULT_MAX_INJECTED_TOKENS, DEFAULT_MIN_RECALL_SCORE,
};
use ::agent::provider::mock::{MockProvider, MockResponse, MockToolCall};
use ::agent::provider::{ProviderError, StreamConfig, StreamEvent, StreamProvider};
use ::agent::types::*;
use ::agent::{AgentContext, AgentTool, ToolContext, ToolError, ToolResult};
use support::{service_with, FakeMemoryBackend};

// ---------------------------------------------------------------------------
// Metered provider — records the size of every request it is handed
// ---------------------------------------------------------------------------

struct MeteredProvider {
    inner: MockProvider,
    sizes: Arc<Mutex<Vec<usize>>>,
}

impl MeteredProvider {
    fn new(inner: MockProvider) -> (Arc<Self>, Arc<Mutex<Vec<usize>>>) {
        let sizes = Arc::new(Mutex::new(Vec::new()));
        (
            Arc::new(Self {
                inner,
                sizes: sizes.clone(),
            }),
            sizes,
        )
    }
}

#[async_trait::async_trait]
impl StreamProvider for MeteredProvider {
    async fn stream(
        &self,
        config: StreamConfig,
        tx: mpsc::UnboundedSender<StreamEvent>,
        cancel: CancellationToken,
    ) -> Result<Message, ProviderError> {
        self.sizes
            .lock()
            .unwrap()
            .push(estimate_request_size(&config.messages));
        self.inner.stream(config, tx, cancel).await
    }
}

/// Cheap token proxy: bytes/4 plus one per message. Good enough to compare two
/// runs of the same case — what matters is the *difference* between them.
fn estimate_request_size(messages: &[Message]) -> usize {
    messages
        .iter()
        .map(|m| {
            let body = match m {
                Message::User { content, .. }
                | Message::Assistant { content, .. }
                | Message::ToolResult { content, .. } => content
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text { text } => Some(text.len()),
                        Content::ToolCall {
                            name, arguments, ..
                        } => Some(name.len() + arguments.to_string().len()),
                        _ => None,
                    })
                    .sum::<usize>(),
            };
            body / 4 + 1
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Scripted case
// ---------------------------------------------------------------------------

struct NoopTool;

#[async_trait::async_trait]
impl AgentTool for NoopTool {
    fn name(&self) -> &str {
        "noop"
    }
    fn label(&self) -> &str {
        "Noop"
    }
    fn description(&self) -> &str {
        "does nothing"
    }
    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}, "required": []})
    }
    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: ToolContext,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult {
            content: vec![Content::Text {
                text: "noop done".into(),
            }],
            details: serde_json::Value::Null,
        })
    }
}

/// The case: ask for a deploy, the model calls the tool, then answers.
fn case_responses() -> Vec<MockResponse> {
    vec![
        MockResponse::ToolCalls(vec![MockToolCall {
            name: "noop".into(),
            arguments: serde_json::json!({}),
            provider_metadata: None,
        }]),
        MockResponse::Text("deployed".into()),
    ]
}

/// Metrics the plan names for the memory-on / memory-off comparison.
#[derive(Debug, Clone)]
struct EvalMetrics {
    /// Time from run start to the first tool execution.
    time_to_first_tool: Option<Duration>,
    /// Time for the whole run.
    end_to_end: Duration,
    /// LLM turns taken.
    turns: u32,
    /// Estimated size of each request the provider received.
    request_sizes: Vec<usize>,
    /// Whether the scripted task reached its expected end state.
    success: bool,
}

impl EvalMetrics {
    fn report(&self, label: &str) {
        println!(
            "  {label:<12} end_to_end={:>6} ms  ttft={:<8} turns={}  requests={:?}  success={}",
            self.end_to_end.as_millis(),
            self.time_to_first_tool
                .map(|d| format!("{}ms", d.as_millis()))
                .unwrap_or_else(|| "none".into()),
            self.turns,
            self.request_sizes,
            self.success,
        );
    }
}

/// Run the scripted case once and collect its metrics.
async fn run_case(
    prep: Option<::agent::PrepareContextFn>,
    context_config: Option<ContextConfig>,
) -> (EvalMetrics, Arc<Mutex<Vec<usize>>>) {
    run_case_with("deploy the release", prep, context_config).await
}

async fn run_case_with(
    prompt: &str,
    prep: Option<::agent::PrepareContextFn>,
    context_config: Option<ContextConfig>,
) -> (EvalMetrics, Arc<Mutex<Vec<usize>>>) {
    let (provider, sizes) = MeteredProvider::new(MockProvider::new(case_responses()));
    let mut config = meter_config(provider as Arc<dyn StreamProvider>);
    config.prepare_context = prep;
    config.context_config = context_config;

    let mut context = AgentContext {
        system_prompt: "eval".into(),
        messages: Vec::new(),
        tools: vec![Box::new(NoopTool)],
    };
    let (tx, mut rx) = mpsc::unbounded_channel();
    let prompt = prompt.to_string();

    let start = Instant::now();
    let join = tokio::spawn(async move {
        agent_loop(
            vec![AgentMessage::Llm(Message::user(prompt))],
            &mut context,
            &config,
            tx,
            CancellationToken::new(),
        )
        .await
    });

    let mut time_to_first_tool = None;
    let mut success = false;
    while let Some(event) = rx.recv().await {
        match event {
            AgentEvent::ToolExecutionStart { tool_name, .. } if tool_name == "noop" => {
                if time_to_first_tool.is_none() {
                    time_to_first_tool = Some(start.elapsed());
                }
            }
            AgentEvent::AgentEnd { messages, .. } => {
                success = messages.iter().any(|m| {
                    matches!(
                        m,
                        AgentMessage::Llm(Message::Assistant {
                            stop_reason: StopReason::Stop,
                            ..
                        })
                    )
                });
            }
            _ => {}
        }
    }

    let new_messages = join.await.expect("loop panicked");
    let turns = new_messages
        .iter()
        .filter(|m| matches!(m, AgentMessage::Llm(Message::Assistant { .. })))
        .count() as u32;

    let request_sizes = sizes.lock().unwrap().clone();
    let metrics = EvalMetrics {
        time_to_first_tool,
        end_to_end: start.elapsed(),
        turns,
        request_sizes,
        success,
    };
    (metrics, sizes)
}

fn meter_config(provider: Arc<dyn StreamProvider>) -> AgentLoopConfig {
    let mut config = AgentLoopConfig::new(provider, "eval-mock");
    config.api_key = "test".into();
    config
}

/// The fact the seeded store holds — deliberately worded to match the query.
const SEEDED_FACT: &str = "the release deploy command is cargo build --release";

// ---------------------------------------------------------------------------
// Memory on vs memory off, same case
// ---------------------------------------------------------------------------

#[tokio::test]
async fn memory_on_versus_memory_off_comparison() {
    println!("\n--- memory on vs memory off (offline fake backend) ---");

    let (off_metrics, _off_sizes) = run_case(None, None).await;
    off_metrics.report("memory off");

    let (svc, backend) = service_with(FakeMemoryBackend::shared(), 16, Duration::from_secs(5));
    backend.seed("m1", SEEDED_FACT, 1.0, None);
    // Make the recall cost observable so the comparison is not decided by
    // sub-millisecond noise on a CI machine.
    backend.set_latency(Duration::from_millis(2));

    let prep = make_prepare_context(svc.clone(), 4, DEFAULT_MAX_INJECTED_TOKENS, None);
    let (on_metrics, _on_sizes) = run_case(Some(prep), None).await;
    on_metrics.report("memory on");

    // Task success is identical: memory never changes what the task means.
    assert!(off_metrics.success, "memory-off arm must succeed");
    assert!(on_metrics.success, "memory-on arm must succeed");

    // Same number of LLM turns — recall costs a round trip, not a turn.
    assert_eq!(on_metrics.turns, off_metrics.turns);
    assert!(on_metrics.turns >= 2, "the case is a two-turn tool run");

    // Telemetry is the observable half of "memory was used".
    let t = svc.telemetry.snapshot();
    assert_eq!(
        t.recall_count, 1,
        "exactly one recall per user request: {t:?}"
    );
    assert!(
        t.injected_tokens > 0,
        "a relevant memory should have been injected: {t:?}"
    );
    assert_eq!(t.errors, 0, "no recall errors: {t:?}");
    assert!(
        t.recall_hits_kept >= 1,
        "the seed was kept, not filtered: {t:?}"
    );

    // The injected block reaches the provider, so the request is larger than
    // the memory-off request — that is the cost being bought.
    let on_max = on_metrics.request_sizes.iter().max().copied().unwrap_or(0);
    let off_max = off_metrics.request_sizes.iter().max().copied().unwrap_or(0);
    assert!(
        on_max > off_max,
        "memory-on request ({on_max}) should exceed memory-off ({off_max})"
    );

    // Block is bounded by the injected-token ceiling.
    assert!(
        (t.injected_tokens as usize) <= DEFAULT_MAX_INJECTED_TOKENS + 32,
        "injected block {} tokens exceeds ceiling {DEFAULT_MAX_INJECTED_TOKENS}",
        t.injected_tokens
    );

    svc.shutdown().await;
}

/// An irrelevant store produces an unmodified request — no padding, no
/// filler hits to fill top_k.
#[tokio::test]
async fn irrelevant_store_injects_nothing() {
    let (svc, backend) = service_with(FakeMemoryBackend::shared(), 16, Duration::from_secs(5));
    backend.seed("m1", "the coffee machine refills at 9am", 1.0, None);
    backend.seed("m2", "the plant on the desk needs watering", 1.0, None);

    let (metrics, _sizes) = run_case(
        Some(make_prepare_context(
            svc.clone(),
            4,
            DEFAULT_MAX_INJECTED_TOKENS,
            None,
        )),
        None,
    )
    .await;

    assert!(metrics.success);
    let t = svc.telemetry.snapshot();
    assert_eq!(
        t.injected_tokens, 0,
        "no relevant memory means no injection, not padding: {t:?}"
    );

    // And the request is identical to the memory-off request.
    let (off, _off_sizes) = run_case(None, None).await;
    assert_eq!(
        metrics.request_sizes, off.request_sizes,
        "an empty recall must produce an unmodified request"
    );

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// Scenario 5 — slow or unavailable embedding leaves the agent operational
// ---------------------------------------------------------------------------

#[tokio::test]
async fn slow_backend_still_completes_with_diagnostics() {
    // Slow enough to matter, servable within the timeout.
    let (svc, backend) = service_with(FakeMemoryBackend::shared(), 16, Duration::from_millis(250));
    backend.seed("m1", SEEDED_FACT, 1.0, None);
    backend.set_latency(Duration::from_millis(30));

    let (metrics, _s) = run_case(
        Some(make_prepare_context(
            svc.clone(),
            4,
            DEFAULT_MAX_INJECTED_TOKENS,
            None,
        )),
        None,
    )
    .await;

    assert!(metrics.success, "the agent must still complete its task");
    let t = svc.telemetry.snapshot();
    assert_eq!(t.errors, 0, "slow is not failed: {t:?}");
    assert!(
        t.injected_tokens > 0,
        "a slow backend still delivered the fact"
    );

    // Unavailable: slower than the timeout, so the caller degrades.
    backend.set_latency(Duration::from_millis(500));

    let (degraded, _s2) = run_case(
        Some(make_prepare_context(
            svc.clone(),
            4,
            DEFAULT_MAX_INJECTED_TOKENS,
            None,
        )),
        None,
    )
    .await;

    assert!(
        degraded.success,
        "an unavailable backend must not break the task"
    );
    let t2 = svc.telemetry.snapshot();
    assert!(t2.errors > 0, "timeout must be visible, not silent: {t2:?}");

    // Hard failure: the backend rejects outright.
    backend.set_broken(true);
    let (broken, _s3) = run_case(
        Some(make_prepare_context(
            svc.clone(),
            4,
            DEFAULT_MAX_INJECTED_TOKENS,
            None,
        )),
        None,
    )
    .await;
    assert!(
        broken.success,
        "an unavailable backend must not break the task"
    );

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// Scenario 6 — near-full context respects token budgeting including memory
// ---------------------------------------------------------------------------

fn long_context_config(budget: usize) -> ContextConfig {
    ContextConfig {
        max_context_tokens: budget,
        system_prompt_tokens: 4,
        keep_recent: 1,
        keep_first: 0,
        tool_output_max_lines: 8,
        tool_output_max_lines_overrides: Default::default(),
        compact_target_ratio: 0.7,
        compact_headroom_turns: Some(30),
        truncate_tool_output_on_append: true,
    }
}

#[tokio::test]
async fn near_full_context_budget_includes_memory() {
    let budget = 120usize;
    // A transcript far over budget: the same deploy question repeated ~300
    // words, so compaction must fire AND the recall must find its match.
    let big_prompt = "how do i deploy the release with cargo build --release ".repeat(6);

    // --- memory off: the request must fit the budget ---
    let (off_metrics, off_sizes) =
        run_case_with(&big_prompt, None, Some(long_context_config(budget))).await;
    let off_max = off_sizes.lock().unwrap().iter().copied().max().unwrap_or(0);
    assert!(off_metrics.success, "the memory-off arm must still run");
    assert!(
        off_max <= budget + 32,
        "memory-off request of {off_max} exceeds budget {budget} — compaction did not hold"
    );

    // --- memory on: the reservation must keep the total within budget ---
    let (svc, backend) = service_with(FakeMemoryBackend::shared(), 16, Duration::from_secs(5));
    // A relevant memory whose block must be *reserved* out of the compaction
    // budget rather than appended on top of it.
    backend.seed("big", SEEDED_FACT, 1.0, None);

    let (on_metrics, on_sizes) = run_case_with(
        &big_prompt,
        Some(make_prepare_context(
            svc.clone(),
            4,
            DEFAULT_MAX_INJECTED_TOKENS,
            None,
        )),
        Some(long_context_config(budget)),
    )
    .await;
    let on_max = on_sizes.lock().unwrap().iter().copied().max().unwrap_or(0);
    let t = svc.telemetry.snapshot();

    assert!(on_metrics.success, "the memory-on arm must still run");
    assert!(
        on_max <= budget + 32,
        "memory-on request of {on_max} exceeds budget {budget} — \
         the prepared block was not reserved, only appended"
    );
    assert!(
        t.injected_tokens > 0,
        "the seeded memory should have been injected: {t:?}"
    );

    svc.shutdown().await;
}

// ---------------------------------------------------------------------------
// Cold initialization is separated from warmed runs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cold_init_separated_from_warm_runs() {
    let backend = FakeMemoryBackend::shared();
    backend.seed("m1", SEEDED_FACT, 1.0, None);
    backend.set_latency(Duration::from_millis(1));

    // --- Cold: constructing the service does no work ---
    let (svc, _b) = service_with(backend.clone(), 16, Duration::from_secs(5));
    assert_eq!(
        backend.dispatched(),
        0,
        "service construction must be lazy — no recall until asked"
    );

    // --- Cold: the first recall pays the backend ---
    let cold = Instant::now();
    let cold_hits = match svc
        .recall(
            "what is the deploy command?",
            4,
            DEFAULT_MIN_RECALL_SCORE,
            None,
        )
        .await
    {
        RecallResult::Hits(h) => h.len(),
        RecallResult::Degraded(r) => panic!("cold recall degraded: {r}"),
    };
    let cold_elapsed = cold.elapsed();
    assert_eq!(cold_hits, 1, "the seeded fact is recalled");
    assert_eq!(backend.dispatched(), 1, "one backend call for one recall");

    // --- Warm: same store, worker already spun up ---
    let warm = Instant::now();
    let warm_hits = match svc
        .recall(
            "what is the deploy command?",
            4,
            DEFAULT_MIN_RECALL_SCORE,
            None,
        )
        .await
    {
        RecallResult::Hits(h) => h.len(),
        RecallResult::Degraded(r) => panic!("warm recall degraded: {r}"),
    };
    let warm_elapsed = warm.elapsed();
    assert_eq!(warm_hits, 1, "warm recall sees the same fact");
    assert_eq!(backend.dispatched(), 2, "two recalls, two backend calls");

    let t = svc.telemetry.snapshot();
    assert_eq!(t.recall_count, 2, "telemetry counts both recalls: {t:?}");

    println!(
        "\n--- cold vs warm (fake backend, 1ms op latency) ---\n  cold = {:?}  warm = {:?}",
        cold_elapsed, warm_elapsed
    );

    svc.shutdown().await;
}
