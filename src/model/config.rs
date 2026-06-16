use crate::gguf::{Error, GgufFile, Result};

/// Model hyperparameters parsed from GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Hidden size / embedding dimension (e.g., 4096).
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Number of key/value heads (GQA); defaults to `n_heads` if absent.
    pub n_kv_heads: usize,
    /// Dimensionality of each head for keys/queries.
    pub head_dim: usize,
    /// Feed-forward intermediate dimension.
    pub intermediate_size: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
    /// RoPE base frequency (theta).
    pub rope_base: f32,
    /// Number of dimensions to apply RoPE to.
    pub rope_dim: usize,
    /// Maximum context length from training.
    pub context_length: usize,
    /// Model architecture string, e.g. "llama".
    pub architecture: String,
}

impl Config {
    /// Parse configuration from standard `llama.*` GGUF metadata keys.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let architecture = file
            .get_string("general.architecture")
            .ok_or_else(|| Error::TensorNotFound("general.architecture".to_string()))?
            .to_string();

        if architecture != "llama" {
            return Err(Error::UnsupportedArchitecture(architecture));
        }

        let hidden_size = get_usize(file, "llama.embedding_length")?;
        let n_layers = get_usize(file, "llama.block_count")?;
        let n_heads = get_usize(file, "llama.attention.head_count")?;
        let n_kv_heads = file
            .get_u32("llama.attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(n_heads);
        let head_dim = file
            .get_u32("llama.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_size / n_heads);
        let intermediate_size = get_usize(file, "llama.feed_forward_length")?;
        let norm_eps = file
            .get_f32("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(1e-5);
        let rope_base = file.get_f32("llama.rope.freq_base").unwrap_or(10_000.0);
        let rope_dim = file
            .get_u32("llama.rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let context_length = file
            .get_u32("llama.context_length")
            .map(|v| v as usize)
            .unwrap_or(4096);
        let vocab_size = file.get_u32("llama.vocab_size").unwrap_or(0);

        Ok(Self {
            vocab_size,
            hidden_size,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate_size,
            norm_eps,
            rope_base,
            rope_dim,
            context_length,
            architecture,
        })
    }
}

fn get_usize(file: &GgufFile, key: &str) -> Result<usize> {
    file.get_u64(key)
        .map(|v| v as usize)
        .ok_or_else(|| Error::TensorNotFound(key.to_string()))
}
