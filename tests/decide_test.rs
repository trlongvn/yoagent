//! CLI decision capability: `agent::decide`.
//!
//! `decide` answers "which of these options matches this input" with one
//! short reply. The library entry points (`decide` / `decide_with_config`)
//! are unit-tested in `src/decide.rs` against `MockProvider`; this file
//! covers what only an end-to-end check can: the loop path the real binary
//! takes (a single `prompt_structured` call through `Agent`), the
//! tool-forcing unwrap, and the guarantee that the answer is always one of
//! the options the caller offered.

use ::agent::decide::{decide, DecisionError};
use ::agent::provider::mock::{MockResponse, MockToolCall};
use ::agent::provider::{MockProvider, ModelConfig};
use ::agent::Agent;

fn opts() -> Vec<String> {
    vec!["cancel".into(), "refund".into(), "other".into()]
}

/// A provider that honours the schema and returns plain JSON text, the way
/// `json_schema` / `responseSchema` providers do.
fn native(answer: &str) -> Agent {
    Agent::from_provider(
        MockProvider::text(format!("\"{answer}\"")),
        ModelConfig::mock(),
    )
}

/// A tool-forcing provider (Anthropic's `native_structured_output` off): the
/// whole payload arrives as a call to the synthetic `structured_output` tool.
fn forced(answer: &str) -> Agent {
    let provider = MockProvider::new(vec![MockResponse::ToolCalls(vec![MockToolCall {
        provider_metadata: None,
        name: "structured_output".into(),
        arguments: serde_json::json!(answer),
    }])]);
    Agent::from_provider(provider, ModelConfig::mock())
}

#[tokio::test]
async fn decides_from_the_native_structured_path() {
    let mut agent = native("refund");
    let d = decide("please refund my last invoice", opts(), &mut agent)
        .await
        .expect("must decide");
    assert_eq!(d.choice, "refund");
    assert_eq!(d.all, opts());
    assert_eq!(d.raw, "refund");
}

#[tokio::test]
async fn decides_from_the_tool_forced_path() {
    // The forced call's arguments are the JSON string payload; unwrapping it
    // is the loop's job, and the answer must survive it.
    let mut agent = forced("cancel");
    let d = decide("cancel my subscription", opts(), &mut agent)
        .await
        .expect("must decide");
    assert_eq!(d.choice, "cancel");
    assert_eq!(d.raw, "cancel");
}

#[tokio::test]
async fn decides_from_free_text_the_schema_was_ignored() {
    // A provider that drops the schema and answers in prose: the payload
    // still has to match one of the options, or the call fails honestly.
    let mut agent = Agent::from_provider(MockProvider::text("other"), ModelConfig::mock());
    let d = decide("question about pricing tiers", opts(), &mut agent)
        .await
        .expect("bare label is a parse error that still matches");
    assert_eq!(d.choice, "other");
    assert_eq!(d.raw, "other");
}

#[tokio::test]
async fn an_unmatched_reply_is_an_error_not_a_guess() {
    let mut agent =
        Agent::from_provider(MockProvider::text("escalate_to_ceo"), ModelConfig::mock());
    let err = decide("chargeback threat", opts(), &mut agent)
        .await
        .expect_err("an invented label must not be returned");
    match err {
        DecisionError::NoMatch { raw } => assert!(raw.contains("escalate_to_ceo"), "{raw}"),
        other => panic!("expected NoMatch, got {other:?}"),
    }
}

#[tokio::test]
async fn a_provider_failure_surfaces_as_an_error() {
    // The default MockProvider sequence runs dry and reports no output.
    let mut agent = Agent::from_provider(MockProvider::new(vec![]), ModelConfig::mock());
    let _ = decide("anything", opts(), &mut agent)
        .await
        .expect_err("must fail");
}

#[tokio::test]
async fn empty_options_are_rejected_before_any_call() {
    let mut agent = native("cancel");
    let err = decide("anything", Vec::new(), &mut agent)
        .await
        .expect_err("no options, no decision");
    assert!(matches!(err, DecisionError::NoOptions), "{err:?}");
}

#[tokio::test]
async fn duplicate_options_are_rejected_before_any_call() {
    let mut agent = native("cancel");
    let err = decide(
        "anything",
        vec!["cancel".into(), "cancel".into()],
        &mut agent,
    )
    .await
    .expect_err("ambiguous options");
    match err {
        DecisionError::DuplicateOptions(dupes) => assert_eq!(dupes, vec!["cancel".to_string()]),
        other => panic!("expected DuplicateOptions, got {other:?}"),
    }
}

/// Records every `StreamConfig` it is handed, so a test can inspect what the
/// decision actually put on the wire.
struct CapturingProvider {
    inner: MockProvider,
    seen: std::sync::Arc<std::sync::Mutex<Vec<::agent::provider::StreamConfig>>>,
}

#[async_trait::async_trait]
impl ::agent::provider::StreamProvider for CapturingProvider {
    async fn stream(
        &self,
        config: ::agent::provider::StreamConfig,
        tx: tokio::sync::mpsc::UnboundedSender<::agent::provider::StreamEvent>,
        cancel: tokio_util::sync::CancellationToken,
    ) -> Result<::agent::types::Message, ::agent::provider::ProviderError> {
        self.seen.lock().unwrap().push(config.clone());
        self.inner.stream(config, tx, cancel).await
    }
}

#[tokio::test]
async fn the_options_reach_the_provider_verbatim() {
    // The schema constrains the provider, but a provider that drops it must
    // still see the labels: capture the outgoing request and check both the
    // schema and the user turn.
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let provider = CapturingProvider {
        inner: MockProvider::text("\"cancel\""),
        seen: seen.clone(),
    };
    let mut agent = Agent::from_provider(provider, ModelConfig::mock());
    let d = decide("terminate the plan", opts(), &mut agent)
        .await
        .expect("must decide");
    assert_eq!(d.choice, "cancel");

    let requests = seen.lock().unwrap();
    assert_eq!(requests.len(), 1, "one decision = one request");
    let schema = requests[0]
        .output_schema
        .as_ref()
        .expect("the decision schema must be threaded to the provider");
    assert_eq!(
        schema.schema["enum"]
            .as_array()
            .expect("enum in the schema"),
        &vec![
            serde_json::json!("cancel"),
            serde_json::json!("refund"),
            serde_json::json!("other"),
        ],
    );
    let turn = requests[0].messages.last().expect("a user turn");
    let text = match turn {
        ::agent::types::Message::User { content, .. } => content
            .iter()
            .filter_map(|c| match c {
                ::agent::types::Content::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>(),
        other => panic!("expected a user message, got {other:?}"),
    };
    assert!(text.contains("terminate the plan"), "{text}");
    assert!(text.contains("cancel, refund, other"), "{text}");
}
