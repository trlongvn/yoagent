//! Config loader — load agent + model configuration from a raw config file.
//!
//! Path comes from the `AGENT_CONFIG` environment variable. Supports
//! JSON and TOML (the latter via the optional `toml` feature). The file
//! deserializes directly into agent's own `ModelConfig` (which already
//! derives `Serialize`/`Deserialize`), so no mapping layer is needed.

use std::env;
use std::path::PathBuf;

use crate::provider::{ApiProtocol, ModelConfig, OpenAiCompat};

/// Memory (uteke) configuration — embedding backend + optional custom model/endpoint.
///
/// Mirrors `uteke_core::EmbeddingSettings`. `backend` selects the embedding
/// engine: `"onnx"` (local, default), `"openai"` (external API), or
/// `"ollama"` (local server). When `backend` is an external/remote one,
/// `model` / `base_url` / `api_key` pick exactly which remote model to use.
#[derive(serde::Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// Embedding backend: "onnx" | "openai" | "ollama".
    pub backend: String,
    /// Embedding model. Empty = backend default (openai→text-embedding-3-small,
    /// ollama→nomic-embed-text, onnx→built-in).
    pub model: String,
    /// Custom endpoint base URL. Empty = backend default.
    pub base_url: String,
    /// API key for OpenAI-compatible backend. Empty = env UTEKE_EMBEDDING_API_KEY.
    pub api_key: String,
    /// Force embedding dimensions. 0 = backend/model default.
    pub dims: usize,
    /// Optional `endpoint_path` override (e.g. "/v1/embeddings").
    pub endpoint_path: String,
    /// Base directory for the memory store. The actual SQLite file and vector
    /// index are placed in a profile subdirectory derived from
    /// `{backend}/{model}/{dims}` so different embedding configurations never
    /// share an index. Default: `"memory"`.
    ///
    /// For backward compatibility, if this points directly to a `.sqlite` file
    /// (legacy `db_path`), it is used as-is without profile isolation.
    pub db_path: String,
}

impl MemoryConfig {
    /// Resolve the effective database path with profile isolation.
    ///
    /// Returns the full path to the SQLite file. When `db_path` ends with
    /// `.sqlite`, it is treated as a legacy explicit path and returned as-is.
    /// Otherwise, a profile subdirectory is appended:
    ///   `{db_path}/{backend}/{model_or_default}/db.sqlite`
    ///
    /// Different embedding models produce incompatible vectors even at the
    /// same dimension count, so the profile includes the model name.
    pub fn resolved_db_path(&self) -> std::path::PathBuf {
        let base = if self.db_path.is_empty() {
            "memory".to_string()
        } else {
            self.db_path.clone()
        };
        // Legacy compat: explicit .sqlite file path → use directly
        if base.ends_with(".sqlite") {
            return std::path::PathBuf::from(base);
        }
        let model_slug = if self.model.is_empty() {
            match self.backend.to_ascii_lowercase().as_str() {
                "openai" => "text-embedding-3-small",
                "ollama" => "nomic-embed-text",
                _ => "default",
            }
        } else {
            &self.model
        };
        let dims_suffix = if self.dims > 0 {
            format!("{}", self.dims)
        } else {
            "auto".to_string()
        };
        std::path::PathBuf::from(&base)
            .join(&self.backend)
            .join(model_slug)
            .join(dims_suffix)
            .join("db.sqlite")
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            backend: "onnx".to_string(),
            model: String::new(),
            base_url: String::new(),
            api_key: String::new(),
            dims: 0,
            endpoint_path: String::new(),
            db_path: "memory".to_string(),
        }
    }
}

/// Raw file format: everything the loader understands. Mirrors the public
/// fields of `ModelConfig`; `compat` and `headers` remain optional.
#[derive(serde::Deserialize, Clone)]
#[serde(default, deny_unknown_fields)]
pub struct AgentConfigFile {
    /// Model identifier sent to the API.
    pub model_id: String,
    /// Human-friendly name.
    pub name: String,
    /// Provider name ("openai", "local", "qwen", ...).
    pub provider: String,
    /// Which API protocol. Defaults to OpenAiCompletions.
    pub api: String,
    /// Base URL without trailing slash.
    pub base_url: String,
    /// Whether reasoning/thinking is supported.
    pub reasoning: bool,
    /// Context window in tokens.
    pub context_window: u32,
    /// Max output tokens.
    pub max_tokens: u32,
    /// OpenAI-compat quirk flags.
    pub compat: Option<String>,
    /// Optional extra headers (e.g. Authorization for gateways).
    pub headers: std::collections::HashMap<String, String>,
    /// Memory (uteke) settings — optional; only used when the `memory` feature
    /// is enabled and you want a non-default embedding backend.
    #[serde(default)]
    pub memory: MemoryConfig,
}

impl Default for AgentConfigFile {
    fn default() -> Self {
        Self {
            model_id: String::new(),
            name: String::new(),
            provider: "openai".into(),
            api: "openai_completions".into(),
            base_url: String::new(),
            reasoning: false,
            context_window: 128_000,
            max_tokens: 4096,
            compat: None,
            headers: Default::default(),
            memory: MemoryConfig::default(),
        }
    }
}

impl AgentConfigFile {
    /// Convert into agent's native `ModelConfig`.
    pub fn into_model_config(self) -> ModelConfig {
        let api = match self.api.as_str() {
            "anthropic" | "anthropic_messages" => ApiProtocol::AnthropicMessages,
            "openai_responses" => ApiProtocol::OpenAiResponses,
            "azure" | "azure_openai_responses" => ApiProtocol::AzureOpenAiResponses,
            "google" | "google_generative_ai" => ApiProtocol::GoogleGenerativeAi,
            "vertex" | "google_vertex" => ApiProtocol::GoogleVertex,
            "bedrock" => ApiProtocol::BedrockConverseStream,
            _ => ApiProtocol::OpenAiCompletions,
        };

        let compat = self
            .compat
            .as_deref()
            .map(|c| match c.to_ascii_lowercase().as_str() {
                "meta" => OpenAiCompat::meta(),
                "xai" => OpenAiCompat::xai(),
                "groq" => OpenAiCompat::groq(),
                "cerebras" => OpenAiCompat::cerebras(),
                "openrouter" => OpenAiCompat::openrouter(),
                "mistral" => OpenAiCompat::mistral(),
                "deepseek" => OpenAiCompat::deepseek(),
                "zai" => OpenAiCompat::zai(),
                "minimax" => OpenAiCompat::minimax(),
                "qwen" => OpenAiCompat::qwen(),
                "ollama" => OpenAiCompat::ollama(),
                _ => OpenAiCompat::openai(),
            });

        let model_id = self.model_id.clone();
        // `ModelConfig` is `#[non_exhaustive]` (private `list_priced`), so it
        // is built through the normal constructor and then configured: the
        // config file carries its own protocol, headers and window, which the
        // constructors do not know about.
        let mut config = ModelConfig::openai_compat(
            self.base_url,
            self.model_id,
            self.provider,
            OpenAiCompat::openai(),
        );
        config.api = api;
        config.name = if self.name.is_empty() {
            model_id
        } else {
            self.name
        };
        config.reasoning = self.reasoning;
        config.context_window = self.context_window;
        config.max_tokens = self.max_tokens;
        config.cost = Default::default();
        config.headers = self.headers;
        config.compat = compat;
        config
    }
}
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("AGENT_CONFIG not set and no default found")]
    MissingEnv,
    #[error("config file not found: {0}")]
    NotFound(PathBuf),
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config: {0}")]
    Parse(#[from] serde_json::Error),
    #[cfg(feature = "config-file")]
    #[error("failed to parse TOML config: {0}")]
    Toml(#[from] toml::de::Error),
}

/// Resolve the config path. Priority:
/// 1. `AGENT_CONFIG` env var (explicit)
/// 2. `./agent.json` (default, for zero-config runs)
pub fn resolve_config_path() -> Result<PathBuf, ConfigError> {
    if let Ok(p) = env::var("AGENT_CONFIG") {
        let path = PathBuf::from(p);
        if !path.exists() {
            return Err(ConfigError::NotFound(path));
        }
        return Ok(path);
    }
    let default = PathBuf::from("agent.json");
    if default.exists() {
        return Ok(default);
    }
    Err(ConfigError::MissingEnv)
}

/// Parse a config file into `AgentConfigFile` (auto-detect JSON vs TOML by extension).
pub fn load_config_raw() -> Result<AgentConfigFile, ConfigError> {
    let path = resolve_config_path()?;
    let text = std::fs::read_to_string(&path).map_err(|source| ConfigError::Read {
        path: path.clone(),
        source,
    })?;
    #[cfg(feature = "config-file")]
    {
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            return Ok(toml::from_str(&text)?);
        }
    }
    Ok(serde_json::from_str(&text)?)
}

/// Load a ready-to-use `ModelConfig` from the env-configured file.
pub fn load_model_config() -> Result<ModelConfig, ConfigError> {
    Ok(load_config_raw()?.into_model_config())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    // Env vars are process-global; serialize tests that touch them.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<R>(name: &str, value: &str, f: impl FnOnce() -> R) -> R {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var(name, value) };
        let r = f();
        unsafe { std::env::remove_var(name) };
        r
    }

    #[test]
    fn json_config_loads() {
        let dir = std::env::temp_dir().join("agent_cfg_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("agent.json");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(
            f,
            r#"{{
                "model_id": "qwen3-0.6b",
                "name": "Qwen Local",
                "provider": "local",
                "api": "openai_completions",
                "base_url": "http://localhost:1234/v1",
                "reasoning": false,
                "context_window": 4096,
                "max_tokens": 1024,
                "compat": "qwen",
                "headers": {{}}
            }}"#
        )
        .unwrap();
        let mc = with_env("AGENT_CONFIG", path.to_str().unwrap(), || {
            load_model_config().unwrap()
        });
        assert_eq!(mc.id, "qwen3-0.6b");
        assert_eq!(mc.base_url, "http://localhost:1234/v1");
        assert_eq!(mc.api, ApiProtocol::OpenAiCompletions);
        assert!(mc.compat.is_some());
    }

    #[test]
    fn missing_env_errors() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("AGENT_CONFIG") };
        let err = load_model_config().unwrap_err();
        assert!(matches!(err, ConfigError::MissingEnv));
    }

    #[cfg(feature = "config-file")]
    #[test]
    fn toml_config_loads() {
        let dir = std::env::temp_dir().join("agent_cfg_test_toml");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("agent.toml");
        let mut f = std::fs::File::create(&path).unwrap();
        write!(
            f,
            r#"model_id = "llama3.1:8b"
name = "Ollama llama"
provider = "ollama"
api = "openai_completions"
base_url = "http://localhost:11434/v1"
reasoning = false
context_window = 8192
max_tokens = 2048
compat = "ollama"
[headers]
"#
        )
        .unwrap();
        let mc = with_env("AGENT_CONFIG", path.to_str().unwrap(), || {
            load_model_config().unwrap()
        });
        assert_eq!(mc.id, "llama3.1:8b");
        assert_eq!(mc.base_url, "http://localhost:11434/v1");
        assert_eq!(mc.api, ApiProtocol::OpenAiCompletions);
    }
}
