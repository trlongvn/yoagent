//! Long-term memory integration demo — agent + Uteke.
//!
//! Uteke (uteke-core) is a blocking SQLite+HNSW store, so memory operations
//! run BEFORE the (async) agent loop starts — never inside the tokio runtime.
//!
//! Run (first run downloads ONNX embedding model ~200MB, one-time):
//!   ORT_LIB_PATH=...onnxruntime.dll cargo run --example memory_integration

use std::sync::Arc;

use ::agent::memory_integration::{MemoryBackend, RecallResult, UtekeMemory};
use ::agent::provider::MockProvider;
use ::agent::types::{AgentEvent, StreamDelta};

fn main() {
    // --- Phase 1 (sync): memory store + store/recall ---
    let memory = Arc::new(UtekeMemory::new("memory/db.sqlite"));

    // Store a few facts (simulate prior sessions)
    for fact in [
        "User prefers responses in Vietnamese",
        "Deploy v2.1 to staging at 3pm",
    ] {
        match memory.remember(fact, &["demo"], None, None, None) {
            agent::memory_integration::RememberResult::Ok(id) => {
                println!("stored: {id}");
            }
            agent::memory_integration::RememberResult::Failed(e) => {
                panic!("remember failed: {e}");
            }
        }
    }

    // Semantic recall (first call loads embedding model)
    let hits = match memory.recall("language preference", 3, 0.3, None) {
        RecallResult::Hits(h) => h,
        RecallResult::Degraded(reason) => {
            eprintln!("recall degraded: {reason}");
            Vec::new()
        }
    };
    println!("=== Recalled memories (uteke semantic) ===");
    println!("  hits: {}", hits.len());
    for h in hits.iter().take(3) {
        println!("  • [{} | {:.2}] {}", h.memory_type, h.score, h.content);
    }

    // --- Phase 2 (async): agent loop with MockProvider ---
    tokio::runtime::Runtime::new()
        .expect("failed to create tokio runtime")
        .block_on(async {
            let provider = MockProvider::text("I recall the relevant prior context.");
            let mut agent =
                ::agent::core::Agent::from_provider(provider, agent::provider::ModelConfig::mock())
                    .with_system_prompt("You are a helpful assistant.");

            println!("\n=== Session with memory hooks ===");
            let mut rx = agent.prompt("What language should I use?").await;
            while let Some(event) = rx.recv().await {
                match event {
                    AgentEvent::MessageUpdate {
                        delta: StreamDelta::Text { delta },
                        ..
                    } => print!("{delta}"),
                    AgentEvent::AgentEnd { .. } => break,
                    _ => {}
                }
            }
            println!("\nDone. Memory store at: memory/db.sqlite");
        });
}
