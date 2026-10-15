//! Load ModelConfig from a raw config file.
//!
//! The file path is read from the `AGENT_CONFIG` environment variable
//! (falling back to `./agent.json`). Supports JSON and TOML.
//!
//! Run:
//!   $env:AGENT_CONFIG = "examples/config/agent.json"
//!   cargo run --example config_from_file

use ::agent::config::{load_config_raw, load_model_config};

fn main() {
    // Raw structural config (for printing / validation)
    let cfg = load_config_raw().expect("failed to load raw config");
    println!("Raw config loaded:");
    println!("  model_id   : {}", cfg.model_id);
    println!("  provider   : {}", cfg.provider);
    println!("  api        : {}", cfg.api);
    println!("  base_url   : {}", cfg.base_url);
    println!("  max_tokens : {}", cfg.max_tokens);

    // Converted native ModelConfig (ready for Agent::from_config)
    let mc = load_model_config().expect("failed to convert to ModelConfig");
    println!("\nModelConfig (native):");
    println!("  id     : {}", mc.id);
    println!("  name   : {}", mc.name);
    println!("  api    : {:?}", mc.api);
    println!("  base_url: {}", mc.base_url);
    println!("  compat : {:?}", mc.compat.is_some());

    // Show how you would wire it into an Agent
    println!("\nUsage: Agent::from_config(mc).with_tools(...)");
}
