//! agent-cli — interactive agent CLI built from the agent library.
//!
//! Configuration via a single raw config file (JSON/TOML), path from the
//! `AGENT_CONFIG` environment variable (or `--config` flag). The same file
//! drives both the LLM provider and the optional Uteke memory backend.
//!
//! Build:
//!   cargo build --release --bin agent-cli --features memory
//!
//! Run:
//!   $env:AGENT_CONFIG = "examples/config/agent.json"
//!   agent-cli.exe
//!
//! Flags:
//!   --config FILE   override AGENT_CONFIG
//!   --no-memory       disable uteke memory even if compiled in
//!   /quit             exit the REPL

use std::io::{self, BufRead, Write};

#[cfg(feature = "memory")]
use std::sync::Arc;

use ::agent::config::load_config_raw;
use ::agent::core::Agent;
use ::agent::tools::default_tools;
use ::agent::*;

#[cfg(feature = "memory")]
use ::agent::memory_integration::UtekeMemory;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // --- Parse flags ---
    let args: Vec<String> = std::env::args().collect();
    let mut config_path: Option<String> = None;
    #[cfg(feature = "memory")]
    let mut use_memory = true;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                config_path = Some(args.get(i).cloned().unwrap_or_default());
            }
            #[cfg(feature = "memory")]
            "--no-memory" => use_memory = false,
            _ => {}
        }
        i += 1;
    }

    // --- Load config from file (env or --config) ---
    if let Some(cp) = config_path {
        unsafe { std::env::set_var("AGENT_CONFIG", &cp) };
    }
    let raw = load_config_raw().unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        eprintln!("Set AGENT_CONFIG=<path> or use --config <file> (JSON/TOML)");
        std::process::exit(1);
    });
    let model_cfg = raw.clone().into_model_config();
    println!("model: {} @ {}", model_cfg.id, model_cfg.base_url);

    // --- Optional Uteke memory from the same config file ---
    #[cfg(feature = "memory")]
    let memory: Option<Arc<UtekeMemory>> = if use_memory {
        match UtekeMemory::from_config(&raw.memory) {
            Ok(m) => {
                println!("memory: uteke ({})", raw.memory.backend);
                Some(Arc::new(m))
            }
            Err(e) => {
                eprintln!("memory disabled: {e}");
                None
            }
        }
    } else {
        None
    };
    #[cfg(not(feature = "memory"))]
    let memory: Option<()> = None;
    #[allow(unused_variables)]
    let _memory = memory;

    // --- Build agent ---
    let mut agent = Agent::from_config(model_cfg)
        .with_system_prompt("You are a helpful assistant with tools. Use them when needed; answer directly otherwise.")
        .with_tools(default_tools());

    // Wire memory into the loop
    #[cfg(feature = "memory")]
    if let Some(mem) = &_memory {
        let top_k = 4;
        // Route the blocking Uteke backend through the bounded worker so it
        // can never stall the async loop, and so both the recall hook and the
        // two tools share one queue.
        let svc = ::agent::memory_integration::MemoryService::new(
            mem.clone(),
            64,
            std::time::Duration::from_secs(5),
        );
        agent = ::agent::memory_integration::install_memory_hooks(agent, svc.clone(), top_k);
        agent = agent.with_tools(vec![
            Box::new(::agent::memory_integration::MemorySaveTool {
                service: svc.clone(),
            }),
            Box::new(::agent::memory_integration::MemoryManageTool { service: svc }),
        ]);
    }

    println!("\n  agent-cli - agent with config-file + memory");
    println!("  Type /quit to exit, /clear to reset\n");

    // --- Interactive REPL ---
    let stdin = io::stdin();

    loop {
        print!("\n>> ");
        io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        match line.as_str() {
            "/quit" | "/exit" => break,
            "/clear" => {
                agent.clear_messages();
                println!("(history cleared)\n");
                continue;
            }
            _ => {}
        }

        let mut rx = agent.prompt(&line).await;
        let mut in_text = false;
        let mut session_stats = SessionStats::default();

        while let Some(event) = rx.recv().await {
            match event {
                AgentEvent::MessageUpdate {
                    delta: StreamDelta::Text { delta },
                    ..
                } => {
                    in_text = true;
                    print!("{delta}");
                    io::stdout().flush().unwrap();
                }
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    if in_text {
                        println!();
                        in_text = false;
                    }
                    println!("  ▶ {tool_name}");
                }
                AgentEvent::AgentEnd { stats, .. } => {
                    session_stats = stats.clone();
                }
                _ => {}
            }
        }
        // Restore agent state (messages, tools) before the next turn
        agent.finish().await;
        if in_text {
            println!();
        }
        print_stats(&session_stats);
    }

    println!("\n  bye 👋\n");
}

fn print_stats(stats: &SessionStats) {
    let u = &stats.usage;
    if u.input == 0 && u.output == 0 && u.cache_read == 0 && u.cache_write == 0 {
        return;
    }
    println!(
        "  tokens: {} in / {} out over {} turn(s)",
        u.input, u.output, stats.turns
    );
    if u.cache_read > 0 || u.cache_write > 0 {
        println!(
            "  cache: {:.1}% hit ({} read, {} write)",
            stats.cache_hit_rate() * 100.0,
            u.cache_read,
            u.cache_write
        );
    }
}
