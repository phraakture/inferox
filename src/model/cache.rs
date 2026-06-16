use crate::model::Config;

/// Key/value cache for autoregressive transformer generation.
///
/// Each layer stores a flat buffer shaped `[max_seq_len, n_kv_heads * head_dim]`.
/// `len` tracks how many tokens are currently cached.
#[derive(Debug)]
pub struct KvCache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub len: usize,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, max_seq_len: usize, config: &Config) -> Self {
        let kv_dim = config.n_kv_heads * config.head_dim;
        Self {
            k: vec![vec![0.0; max_seq_len * kv_dim]; n_layers],
            v: vec![vec![0.0; max_seq_len * kv_dim]; n_layers],
            len: 0,
            max_seq_len,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Return the current cached sequence length.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Scratch buffers sized for one-token decode steps.
///
/// Unlike `LayerBuffers`, these are sized for `seq_len = 1` but reuse the same
/// memory across every decode step.
#[derive(Debug)]
pub struct DecodeBuffers {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub scores: Vec<f32>,
    pub head_out: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_gate: Vec<f32>,
    pub ffn_up: Vec<f32>,
    pub ffn_mid: Vec<f32>,
    pub ffn_down: Vec<f32>,
    pub residual: Vec<f32>,
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

impl DecodeBuffers {
    pub fn new(max_seq_len: usize, config: &Config) -> Self {
        let vocab_size = config.vocab_size as usize;
        let hidden_size = config.hidden_size;
        let kv_dim = config.n_kv_heads * config.head_dim;
        let intermediate_size = config.intermediate_size;

        Self {
            q: vec![0.0; hidden_size],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            scores: vec![0.0; max_seq_len],
            head_out: vec![0.0; config.head_dim],
            attn_out: vec![0.0; hidden_size],
            ffn_gate: vec![0.0; intermediate_size],
            ffn_up: vec![0.0; intermediate_size],
            ffn_mid: vec![0.0; intermediate_size],
            ffn_down: vec![0.0; hidden_size],
            residual: vec![0.0; hidden_size],
            hidden: vec![0.0; hidden_size],
            logits: vec![0.0; vocab_size],
        }
    }
}
