// Enables the `doc_cfg` feature badges on docs.rs only; a no-op on stable.
#![cfg_attr(docsrs, feature(doc_cfg))]

//! **agent** â€” the agent runtime for Rust.
//!
//! A simple, effective agent loop with tool execution and event streaming:
//! `Prompt â†’ LLM stream â†’ tool execution â†’ loop`. The loop is the product;
//! everything else is optional layers on top of it.
//!
//! # Quick start
//!
//! ```no_run
//! use agent::{Agent, provider::ModelConfig, tools};
//!
//! # #[tokio::main]
//! # async fn main() {
//! // Provider is selected from the config's protocol; the API key is read
//! // from ANTHROPIC_API_KEY. Call `.with_api_key(...)` to override.
//! let mut agent = Agent::from_config(ModelConfig::anthropic("claude-sonnet-5", "Sonnet 5"))
//!     .with_system_prompt("You are a helpful coding assistant.")
//!     .with_tools(tools::default_tools());
//!
//! let mut events = agent.prompt("List the files in the current directory").await;
//! while let Some(event) = events.recv().await {
//!     // stream text deltas, tool calls, usage â€” render however you like
//! }
//! agent.finish().await;
//! # }
//! ```
//!
//! # What's in the box
//!
//! - **The loop** ([`agent_loop()`](agent_loop())) â€” a stateless free function; [`Agent`] is an
//!   optional stateful wrapper (history, tool registry, steering queues).
//! - **7 provider protocols, 20+ providers** ([`provider`]) â€” Anthropic,
//!   OpenAI (Completions + Responses), Azure, Gemini, Vertex, Bedrock, plus
//!   OpenAI-compatible gateways (Groq, DeepSeek, xAI, OpenCode, Ollama, ...)
//!   with per-provider quirk flags.
//! - **Tools** ([`tools`]) â€” bash, read/write/edit file, search; add your own
//!   via the [`AgentTool`] trait. [MCP](mcp) servers and
//!   [OpenAPI specs](https://docs.rs/yoagent/latest/yoagent/openapi/index.html)
//!   (feature `openapi`) become tools transparently.
//! - **Steering** â€” inject guidance into a running agent ([`Agent::steer`]);
//!   picked up between tool executions (per batch under the default parallel
//!   strategy). Queue follow-ups, inspect/edit the queues.
//! - **Structured outputs** ([`Agent::prompt_structured`]) â€” typed,
//!   schema-validated replies, enforced natively where supported (forced
//!   tool call / `json_schema` / `responseSchema`).
//! - **Permissions** ([`ToolMiddleware`]) â€” async approve/deny/modify hooks
//!   gating every tool call; the mechanism behind approval prompts and
//!   policy engines (agent ships no policy â€” you install it).
//! - **Input filtering** ([`InputFilter`]) â€” rewrite or reject user input
//!   before it reaches the model (PII redaction, prompt-injection guards).
//! - **Sub-agents** ([`SubAgentTool`]) â€” delegation with per-sub-agent models
//!   and [`SharedState`] for passing artifacts by reference.
//! - **GASP** (feature `gasp`) â€” record runs into a
//!   [GASP](https://github.com/yologdev/gasp) agent repo: append-only
//!   semantic event log, restore = clone + replay, conformance-checked in CI.
//! - **Session trees** ([`Session`]) â€” branching conversation history with
//!   fork, checkpoints, and JSONL persistence; edit an earlier turn and
//!   re-run without losing the original branch.
//! - **Context management** ([`context`]) â€” token tracking and tiered
//!   compaction so long sessions keep running.
//! - **Skills** ([`skills`]) â€” load `SKILL.md` files per the
//!   [AgentSkills](https://agentskills.io) standard.
//! - **Telemetry** â€” `tracing` spans per loop/LLM-stream/tool with token and
//!   cost fields; bridge to OpenTelemetry app-side, negligible cost otherwise.
//! - **Testing** ([`provider::mock::MockProvider`]) â€” script a whole multi-turn
//!   tool-calling conversation with no network. It honours the cancellation
//!   token, so abort and steering paths are testable too.
//!
//! The [book](https://yologdev.github.io/yoagent/) covers concepts and
//! provider-specific guides.

pub mod agent_loop;
pub mod context;
pub mod core;
pub mod decide;
pub mod llm_compaction;
pub mod mcp;
pub mod provider;
pub mod retry;
pub mod session;
pub mod shared_state;
pub mod skills;
pub mod sub_agent;
pub mod tools;
pub mod types;

#[cfg(feature = "openapi")]
#[cfg_attr(docsrs, doc(cfg(feature = "openapi")))]
pub mod openapi;

#[cfg(feature = "gasp")]
#[cfg_attr(docsrs, doc(cfg(feature = "gasp")))]
pub mod gasp;

pub use agent_loop::{agent_loop, agent_loop_continue, PrepareContextFn};
pub use context::{CompactionStrategy, DefaultCompaction};
pub use core::{Agent, AgentBuildError, StructuredPromptError};
pub use decide::{decide, Decision, DecisionError};
pub use llm_compaction::LlmCompaction;
pub use retry::RetryConfig;
pub use session::{Session, SessionEntry, SessionError};
pub use shared_state::SharedState;
pub use skills::SkillSet;
pub use sub_agent::SubAgentTool;
pub use types::*;

#[cfg(any(feature = "memory", feature = "memory-onnx"))]
pub mod memory_integration;

pub mod tui;

pub mod config;
