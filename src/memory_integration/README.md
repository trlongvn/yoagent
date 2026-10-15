# Agent Memory Integration — Uteke (Builtin)

Local-first semantic memory for this agent, now a **builtin** behind the
`memory` cargo feature, configured from the **same config file** as the model.

## Cargo features

```toml
[features]
memory = ["dep:uteke-core", "config-file"]   # API: agent::memory_integration
config-file = ["dep:toml"]                   # TOML support in config loader
```

- `cargo build` — no memory (uteke omitted entirely)
- `cargo build --features memory` — uteke-core compiled in

## Single config file drives LLM + memory

Point `AGENT_CONFIG` at one file; the `[memory]` section selects the
embedding backend:

```json
{
  "model_id": "qwen3-0.6b",
  "provider": "local",
  "base_url": "http://localhost:1234/v1",
  "memory": {
    "backend": "onnx",                       // onnx | openai | ollama
    "model": "",                             // "" = backend default
    "base_url": "",
    "api_key": "",
    "dims": 0,
    "endpoint_path": "",
    "db_path": "memory/db.sqlite"
  }
}
```

### Embedding backends

| backend | uses | when |
|---|---|---|
| `onnx`   | bundled ONNX model (~207MB, offline) | default, fully offline |
| `openai` | external OpenAI-compatible API        | higher quality, cloud |
| `ollama` | local Ollama `/api/embed`            | already running Ollama |

### External LLM (OpenAI) embedding — authenticated

API key resolution order (uteke): `[memory].api_key` in config → env
`UTEKE_EMBEDDING_API_KEY` → env `OPENAI_API_KEY`.

```json
{
  "memory": {
    "backend": "openai",
    "model": "text-embedding-3-small",
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-REAL-KEY",            // or leave empty + set env
    "dims": 1536,
    "db_path": "memory/db-openai.sqlite" // separate DB per backend
  }
}
```

> Use a **distinct `db_path` per backend** — a DB's vector index is
> dimension-locked. Since `resolved_db_path()` derives
> `{db_path}/{backend}/{model}/{dims}/db.sqlite`, a default `db_path` of
> `memory` already isolates each backend/model pair, and a legacy explicit
> `.sqlite` path is used verbatim for backward compatibility. Changing the
> model *within* the same dimension count still rebuilds into its own
> profile directory, because vectors from different models are not
> interchangeable even at identical dimensionality.

## Rust API

```rust
use agent::config::{load_model_config, load_config_raw};
use agent::memory_integration::UtekeMemory;

let model = load_model_config()?;                    // LLM from config
let mem_cfg = load_config_raw()?.memory;             // memory from config
let memory = UtekeMemory::from_config(&mem_cfg)?;    // uteke with chosen backend

memory.remember("User prefers Vietnamese", &["pref"], None, Some("note"), None)?;
memory.recall("what language?", 3, 0.3, None);
```

## Wiring into the loop

`UtekeMemory` is a blocking backend, so it must never run on the loop's async
task. `MemoryService` puts it behind a bounded queue plus a dedicated
`memory-worker` thread with a per-op timeout, and exposes:

- `recall(query, top_k, min_score, namespace)` — `Hits` or `Degraded`
- `remember(content, tags, namespace, memory_type, metadata)`
- `get(id)`, `forget(id)`, `soft_forget(id, reason)`, `list(namespace, limit)`
- `telemetry` — `MemoryTelemetrySnapshot` counters (recall/remember counts,
  latency, hits before/after filtering, injected tokens, errors)

The standard wiring, used by `agent-cli` and `agent-tui`:

```rust
let svc = MemoryService::new(mem.clone(), 64, Duration::from_secs(5));

// Recall runs through `prepare_context`: once per new user request, before
// the compaction budget is computed, so its size counts against that budget.
let agent = install_memory_hooks(agent, svc.clone(), top_k);

// Selective writes: the assistant may end a turn with "remember: <fact>" or
// "forget: <query>"; the turn's stop reason decides whether either runs.
let agent = agent.with_tools(vec![
    Box::new(MemorySaveTool { service: svc.clone() }),
    Box::new(MemoryManageTool { service: svc }),
]);
```

`install_memory_hooks_scoped(agent, service, top_k, namespace)` restricts both
recall and writes to one namespace.

### Two rules the hooks enforce

- **No false success memories.** An aborted or errored turn writes nothing.
  Recall blocks are framed as reference data with an explicit statement that
  they are *not* instructions and do not override the current request.
- **Relevance over fill.** Below `DEFAULT_MIN_RECALL_SCORE` nothing is
  injected — an empty block, not a padded one. `top_k` is a ceiling, not a
  quota. `DEDUP_THRESHOLD` (0.92) supersedes only *restatements* of an
  existing fact in the same namespace; a genuinely changed value is left
  alone and must be removed explicitly (`forget:` or the manage tool).

## Cache-safe injection

Dynamic memory is appended AFTER system+tools so prompt-prefix caching stays
intact, and via `prepare_context` so it lands on the outgoing request copy
only — never in the stored transcript, so repeated tool iterations cannot
grow the context by the block's size.

## Runtime requirements

- `onnx` backend needs ONNX Runtime DLL: `ORT_LIB_PATH`.
- `openai` backend needs a real API key.
- `ollama` backend needs Ollama running locally.
