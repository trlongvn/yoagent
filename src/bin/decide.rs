//! decide — pick one option from a list, given an input.
//!
//! Reads an input and a set of candidate options, asks the model which one
//! fits, and prints **only the chosen option** on stdout. Everything else
//! (errors, diagnostics) goes to stderr, so the command composes in a shell
//! pipeline or a `$(...)` capture.
//!
//! Build:
//!   cargo build --release --bin decide
//!
//! Run (config file drives the provider, same as agent-cli):
//!   $env:AGENT_CONFIG = "examples/config/agent.json"
//!   decide --input "cancel my subscription" --options "cancel,refund,other"
//!   # -> cancel
//!
//!   # compose in a pipeline — stdout carries only the choice
//!   kind=$(decide --input "$text" --options "bug,feature,question")
//!
//! Flags:
//!   --input TEXT      the input to classify (required, or piped on stdin)
//!   --options <a,b,c>   comma-separated options (required)
//!   --config FILE     override AGENT_CONFIG
//!   --json              print the full Decision as JSON instead of the bare choice
//!   --stdin             read the input from stdin (default when --input is absent)
//!   -h, --help          usage

use std::io::{self, Read};

use ::agent::config::load_config_raw;
use ::agent::decide::{decide, Decision, DecisionError};
use ::agent::provider::ProviderRegistry;
use ::agent::provider::{ApiProtocol, ModelConfig};

const USAGE: &str = "\
decide — pick one option from a list, given an input

USAGE:
    decide --input <text> --options <a,b,c> [--config <file>] [--json]
    echo \"<text>\" | decide --options <a,b,c>

FLAGS:
    --input <text>      the input to classify. If omitted, read stdin.
    --options <a,b,c>   comma-separated candidate options (required)
    --config <file>     config file to use (default: $AGENT_CONFIG)
    --json              print the full Decision as JSON, not just the choice
    --stdin             force reading the input from stdin
    -h, --help          show this message

Output:
    stdout  the chosen option, verbatim, and nothing else
    stderr  errors and diagnostics
";

struct Args {
    input: Option<String>,
    options: Vec<String>,
    config: Option<String>,
    json: bool,
    help: bool,
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut args = Args {
        input: None,
        options: Vec::new(),
        config: None,
        json: false,
        help: false,
    };

    let mut i = 0;
    while i < argv.len() {
        let arg = argv[i].as_str();
        match arg {
            "-h" | "--help" => args.help = true,
            "--json" => args.json = true,
            "--stdin" => args.input = None,
            "--input" | "-i" => {
                i += 1;
                let v = argv
                    .get(i)
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                args.input = Some(v.clone());
            }
            "--options" | "-o" => {
                i += 1;
                let v = argv
                    .get(i)
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                args.options = parse_options(v);
            }
            "--config" | "-c" => {
                i += 1;
                let v = argv
                    .get(i)
                    .ok_or_else(|| format!("{arg} requires a value"))?;
                args.config = Some(v.clone());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    Ok(args)
}

/// Split a comma-separated option list, trimming whitespace and dropping
/// empty entries — so `"a, b ,, c"` yields `[a, b, c]` rather than a blank
/// option the model could be asked to pick.
fn parse_options(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

fn read_stdin() -> Result<String, String> {
    let mut buf = String::new();
    io::stdin()
        .read_to_string(&mut buf)
        .map(|_| buf)
        .map_err(|e| format!("cannot read stdin: {e}"))
}

// The crate depends on tokio's `rt` (single-threaded) only — `rt-multi-thread`
// is not a dependency, so `#[tokio::main]`'s default flavor would not compile.
// One decision is one short request; there is nothing to parallelise.
#[tokio::main(flavor = "current_thread")]
async fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    if args.help {
        print!("{USAGE}");
        return;
    }

    let input = match args.input {
        Some(i) => i,
        None => match read_stdin() {
            Ok(s) => s.trim().to_owned(),
            Err(e) => {
                eprintln!("error: {e}\n\n{USAGE}");
                std::process::exit(2);
            }
        },
    };

    if args.options.is_empty() {
        eprintln!("error: --options is required\n\n{USAGE}");
        std::process::exit(2);
    }

    if input.trim().is_empty() {
        eprintln!("error: input is empty (pass --input or pipe on stdin)");
        std::process::exit(2);
    }

    // Load the provider config, same file as agent-cli drives.
    if let Some(cp) = &args.config {
        // `set_var` is unsafe in edition 2024; this crate is edition 2021.
        unsafe { std::env::set_var("AGENT_CONFIG", cp) };
    }
    let raw = match load_config_raw() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("config error: {e}");
            eprintln!("Set AGENT_CONFIG=<path> or use --config <file>");
            std::process::exit(2);
        }
    };
    let model_cfg: ModelConfig = raw.into_model_config();
    let registry = ProviderRegistry::default();

    // A decision needs a provider that speaks structured output. Needle 3 is
    // not registered by default (it needs weights on disk), so say so plainly
    // rather than failing with "no provider for protocol".
    if model_cfg.api == ApiProtocol::Needle3 && !registry.has(&ApiProtocol::Needle3) {
        eprintln!(
            "error: model {} uses the needle3 protocol, which needs weights on disk.\n\
             Register it first (ProviderRegistry::register_needle3) or point AGENT_CONFIG \
             at a model with structured-output support.",
            model_cfg.id
        );
        std::process::exit(2);
    }

    let mut agent = match ::agent::core::Agent::from_config_with(&registry, model_cfg) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let decision = decide(&input, args.options.clone(), &mut agent).await;

    match decision {
        Ok(d) => print_decision(&d, args.json),
        Err(DecisionError::NoOptions) => {
            eprintln!("error: at least one option is required");
            std::process::exit(2);
        }
        Err(DecisionError::DuplicateOptions(dupes)) => {
            eprintln!("error: duplicate options: {dupes:?}");
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

/// Print the decision: the bare choice by default, full JSON on `--json`.
/// Both go to stdout so the choice is capturable in a pipeline.
fn print_decision(d: &Decision, as_json: bool) {
    if as_json {
        match serde_json::to_string(d) {
            Ok(j) => println!("{j}"),
            Err(e) => {
                eprintln!("error: cannot serialize decision: {e}");
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", d.choice);
    }
}
