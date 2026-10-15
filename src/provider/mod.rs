pub mod anthropic;
pub mod azure_openai;
pub mod bedrock;
pub mod google;
pub mod google_vertex;
pub mod mock;
pub mod model;
#[cfg(feature = "needle3")]
pub mod needle3;
pub mod openai_compat;
pub mod openai_responses;
pub mod registry;
pub mod sse;
pub mod traits;

pub use anthropic::AnthropicProvider;
pub use azure_openai::AzureOpenAiProvider;
pub use bedrock::BedrockProvider;
pub use google::GoogleProvider;
pub use google_vertex::GoogleVertexProvider;
pub use mock::MockProvider;
pub use model::{AnthropicCompat, ApiProtocol, ContextTier, CostConfig, ModelConfig, OpenAiCompat};
#[cfg(feature = "needle3")]
pub use needle3::{Needle3Outcome, Needle3Provider};
pub use openai_compat::OpenAiCompatProvider;
pub use openai_responses::OpenAiResponsesProvider;
pub(crate) use registry::resolve_api_key_or_warn;
pub use registry::{resolve_api_key, ProviderRegistry};
pub use traits::*;
