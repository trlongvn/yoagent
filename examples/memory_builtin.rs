//! Builtin memory demo — Uteke memory configured from the SAME config file
//! as the model (via `AGENT_CONFIG` env var).
//!
//! The `[memory]` section of the config picks the embedding backend:
//!   - "onnx"   → bundled local ONNX (offline, default)
//!   - "ollama" → local Ollama server (custom model via config)
//!   - "openai" → external LLM API for embeddings (custom model/base_url/api_key)
//!
//! Also loads the model itself from the same file (`load_model_config`),
//! so ONE file drives both the LLM provider and the memory backend.
//!
//! Build & run:
//!   cargo build --features memory --example memory_builtin
//!   $env:AGENT_CONFIG = "examples/config/agent-ollama-embed.json"
//!   cargo run --features memory --example memory_builtin
//!
//! For the openai embedding backend you also need:
//!   $env:UTEKE_EMBEDDING_API_KEY = "sk-..."   (or put it in config api_key)

use std::sync::Arc;

use ::agent::config::{load_model_config, MemoryConfig};
use ::agent::memory_integration::{MemoryBackend, UtekeMemory};

fn main() {
    // --- 1. Load model + memory config from the SAME file ---
    let model_cfg = load_model_config().expect("failed to load model config from AGENT_CONFIG");
    println!(
        "Model (from config): id={} base_url={}",
        model_cfg.id, model_cfg.base_url
    );

    // The loader exposes the raw memory section too (for this demo we read
    // it directly from a re-parse; in production use the config module).
    let raw = agent::config::load_config_raw().expect("load raw config");
    let memory_cfg: MemoryConfig = raw.memory;

    println!(
        "Memory (from config): backend={} model={} base_url={}",
        memory_cfg.backend,
        memory_cfg.model,
        if memory_cfg.base_url.is_empty() {
            "(default)"
        } else {
            &memory_cfg.base_url
        }
    );

    // --- 2. Open Uteke with the configured embedding backend ---
    let memory =
        Arc::new(UtekeMemory::from_config(&memory_cfg).expect("failed to open uteke from config"));
    println!("Uteke opened (backend: {})", memory_cfg.backend);

    // --- 3. Exercise remember / recall ---
    for fact in [
        "User prefers Vietnamese responses",
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

    let hits = match memory.recall("what language does the user prefer?", 3, 0.3, None) {
        agent::memory_integration::RecallResult::Hits(h) => h,
        agent::memory_integration::RecallResult::Degraded(reason) => {
            eprintln!("recall degraded: {reason}");
            Vec::new()
        }
    };
    println!("\n=== Semantic recall ===");
    println!("hits: {}", hits.len());
    for h in hits.iter().take(3) {
        println!("  • [{} | {:.2}] {}", h.memory_type, h.score, h.content);
    }

    println!("\nDone — memory store: memory/db.sqlite");
}
