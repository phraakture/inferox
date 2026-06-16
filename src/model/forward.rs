//! Transformer layer forward pass.

use crate::gguf::Result;
use crate::model::{Config, Layer, Model};
use crate::ops::{add_in_place, matmul_t, rms_norm, rope, softmax_rows, swiglu};

/// Reusable scratch buffers for a single layer forward pass.
#[derive(Debug)]
pub struct LayerBuffers {
    /// `[seq_len, hidden_size]` query projection.
    pub q: Vec<f32>,
    /// `[seq_len, n_kv_heads * head_dim]` key projection.
    pub k: Vec<f32>,
    /// `[seq_len, n_kv_heads * head_dim]` value projection.
    pub v: Vec<f32>,
    /// `[seq_len, seq_len]` attention scores for one head.
    pub scores: Vec<f32>,
    /// `[seq_len, head_dim]` attention output for one head.
    pub head_out: Vec<f32>,
    /// `[seq_len, hidden_size]` concatenated attention head outputs.
    pub attn_out: Vec<f32>,
    /// `[seq_len, intermediate_size]` FFN gate projection.
    pub ffn_gate: Vec<f32>,
    /// `[seq_len, intermediate_size]` FFN up projection.
    pub ffn_up: Vec<f32>,
    /// `[seq_len, intermediate_size]` SwiGLU output.
    pub ffn_mid: Vec<f32>,
    /// `[seq_len, hidden_size]` FFN down projection.
    pub ffn_down: Vec<f32>,
    /// `[seq_len, hidden_size]` residual copy.
    pub residual: Vec<f32>,
}

impl LayerBuffers {
    /// Allocate buffers sized for the given sequence length and model config.
    pub fn new(seq_len: usize, config: &Config) -> Self {
        Self {
            q: vec![0.0; seq_len * config.hidden_size],
            k: vec![0.0; seq_len * config.n_kv_heads * config.head_dim],
            v: vec![0.0; seq_len * config.n_kv_heads * config.head_dim],
            scores: vec![0.0; seq_len * seq_len],
            head_out: vec![0.0; seq_len * config.head_dim],
            attn_out: vec![0.0; seq_len * config.hidden_size],
            ffn_gate: vec![0.0; seq_len * config.intermediate_size],
            ffn_up: vec![0.0; seq_len * config.intermediate_size],
            ffn_mid: vec![0.0; seq_len * config.intermediate_size],
            ffn_down: vec![0.0; seq_len * config.hidden_size],
            residual: vec![0.0; seq_len * config.hidden_size],
        }
    }
}

impl Layer {
    /// Run one transformer layer forward pass in-place on `x`.
    ///
    /// `x` has shape `[seq_len, hidden_size]`. The output is written back into
    /// `x`. This implementation processes the full sequence; KV-cache is not
    /// used.
    pub fn forward(&self, model: &Model, x: &mut [f32], buf: &mut LayerBuffers) -> Result<()> {
        let cfg = &model.config;
        let seq_len = x.len() / cfg.hidden_size;
        let kv_group_size = cfg.n_heads / cfg.n_kv_heads;

        // --- Attention branch ---
        buf.residual.copy_from_slice(x);

        let attn_norm = model.weights.tensor_f32_by_index(self.attn_norm)?;
        rms_norm(x, &attn_norm, cfg.norm_eps);

        let w_q = model.weights.tensor_f32_by_index(self.attn_q)?;
        let w_k = model.weights.tensor_f32_by_index(self.attn_k)?;
        let w_v = model.weights.tensor_f32_by_index(self.attn_v)?;

        // q, k, v projections
        matmul_t(x, &w_q, &mut buf.q, seq_len, cfg.hidden_size, cfg.hidden_size);
        matmul_t(
            x,
            &w_k,
            &mut buf.k,
            seq_len,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.hidden_size,
        );
        matmul_t(
            x,
            &w_v,
            &mut buf.v,
            seq_len,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.hidden_size,
        );

        // Apply RoPE to q and k
        rope(&mut buf.q, seq_len, cfg.n_heads, cfg.head_dim, cfg.rope_dim, cfg.rope_base);
        rope(
            &mut buf.k,
            seq_len,
            cfg.n_kv_heads,
            cfg.head_dim,
            cfg.rope_dim,
            cfg.rope_base,
        );

        // Multi-head attention (with GQA support)
        buf.attn_out.fill(0.0);

        for h_q in 0..cfg.n_heads {
            let h_kv = h_q / kv_group_size;

            for pos in 0..seq_len {
                // q slice for this head and position
                let q_offset = (pos * cfg.n_heads + h_q) * cfg.head_dim;
                let q_head = &buf.q[q_offset..q_offset + cfg.head_dim];

                for t in 0..seq_len {
                    let k_offset = (t * cfg.n_kv_heads + h_kv) * cfg.head_dim;
                    let k_head = &buf.k[k_offset..k_offset + cfg.head_dim];

                    let mut score = 0.0f32;
                    for d in 0..cfg.head_dim {
                        score += q_head[d] * k_head[d];
                    }
                    buf.scores[pos * seq_len + t] = score / (cfg.head_dim as f32).sqrt();
                }
            }

            // Softmax over the key dimension for this query head
            softmax_rows(&mut buf.scores, seq_len, seq_len);

            // Weighted sum of values
            for pos in 0..seq_len {
                let out_offset = pos * cfg.head_dim;
                buf.head_out[out_offset..out_offset + cfg.head_dim].fill(0.0);

                for t in 0..seq_len {
                    let score = buf.scores[pos * seq_len + t];
                    let v_offset = (t * cfg.n_kv_heads + h_kv) * cfg.head_dim;
                    let v_head = &buf.v[v_offset..v_offset + cfg.head_dim];

                    for (d, v_val) in v_head.iter().enumerate().take(cfg.head_dim) {
                        buf.head_out[out_offset + d] += score * v_val;
                    }
                }

                // Scatter into attn_out
                let attn_offset = pos * cfg.hidden_size + h_q * cfg.head_dim;
                buf.attn_out[attn_offset..attn_offset + cfg.head_dim]
                    .copy_from_slice(&buf.head_out[out_offset..out_offset + cfg.head_dim]);
            }
        }

        // Output projection
        let w_o = model.weights.tensor_f32_by_index(self.attn_output)?;
        matmul_t(
            &buf.attn_out,
            &w_o,
            x,
            seq_len,
            cfg.hidden_size,
            cfg.hidden_size,
        );
        add_in_place(x, &buf.residual);

        // --- FFN branch ---
        buf.residual.copy_from_slice(x);

        let ffn_norm = model.weights.tensor_f32_by_index(self.ffn_norm)?;
        rms_norm(x, &ffn_norm, cfg.norm_eps);

        let w_gate = model.weights.tensor_f32_by_index(self.ffn_gate)?;
        let w_up = model.weights.tensor_f32_by_index(self.ffn_up)?;
        let w_down = model.weights.tensor_f32_by_index(self.ffn_down)?;

        matmul_t(
            x,
            &w_gate,
            &mut buf.ffn_gate,
            seq_len,
            cfg.intermediate_size,
            cfg.hidden_size,
        );
        matmul_t(
            x,
            &w_up,
            &mut buf.ffn_up,
            seq_len,
            cfg.intermediate_size,
            cfg.hidden_size,
        );
        swiglu(&buf.ffn_gate, &buf.ffn_up, &mut buf.ffn_mid);
        matmul_t(
            &buf.ffn_mid,
            &w_down,
            &mut buf.ffn_down,
            seq_len,
            cfg.hidden_size,
            cfg.intermediate_size,
        );

        x.copy_from_slice(&buf.ffn_down);
        add_in_place(x, &buf.residual);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{TensorType, ValueType};
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::fs;
    use std::io::Write;

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.write_u64::<LittleEndian>(s.len() as u64).unwrap();
        buf.write_all(s.as_bytes()).unwrap();
    }

    fn write_metadata_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        write_string(buf, key);
        buf.write_u32::<LittleEndian>(ValueType::Uint32 as u32).unwrap();
        buf.write_u32::<LittleEndian>(value).unwrap();
    }

    fn write_metadata_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
        write_string(buf, key);
        buf.write_u32::<LittleEndian>(ValueType::Float32 as u32).unwrap();
        buf.write_f32::<LittleEndian>(value).unwrap();
    }

    fn write_metadata_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_string(buf, key);
        buf.write_u32::<LittleEndian>(ValueType::String as u32).unwrap();
        write_string(buf, value);
    }

    struct TestGgufBuilder {
        metadata: Vec<u8>,
        n_kv: u64,
        tensors: Vec<(String, Vec<u64>, Vec<u8>)>,
    }

    impl TestGgufBuilder {
        fn new() -> Self {
            Self {
                metadata: Vec::new(),
                n_kv: 0,
                tensors: Vec::new(),
            }
        }

        fn metadata_u32(&mut self, key: &str, value: u32) -> &mut Self {
            write_metadata_u32(&mut self.metadata, key, value);
            self.n_kv += 1;
            self
        }

        fn metadata_f32(&mut self, key: &str, value: f32) -> &mut Self {
            write_metadata_f32(&mut self.metadata, key, value);
            self.n_kv += 1;
            self
        }

        fn metadata_string(&mut self, key: &str, value: &str) -> &mut Self {
            write_metadata_string(&mut self.metadata, key, value);
            self.n_kv += 1;
            self
        }

        fn tensor_f32(&mut self, name: &str, shape: &[u64], data: &[f32]) -> &mut Self {
            self.tensors.push((name.to_string(), shape.to_vec(), {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for &v in data {
                    bytes.write_f32::<LittleEndian>(v).unwrap();
                }
                bytes
            }));
            self
        }

        fn write(&self, path: &std::path::Path) {
            let mut buf = Vec::new();
            buf.write_all(b"GGUF").unwrap();
            buf.write_u32::<LittleEndian>(3).unwrap();
            buf.write_u64::<LittleEndian>(self.tensors.len() as u64).unwrap();
            buf.write_u64::<LittleEndian>(self.n_kv).unwrap();
            buf.extend_from_slice(&self.metadata);

            for (name, shape, _data) in &self.tensors {
                write_string(&mut buf, name);
                buf.write_u32::<LittleEndian>(shape.len() as u32).unwrap();
                for &dim in shape {
                    buf.write_u64::<LittleEndian>(dim).unwrap();
                }
                buf.write_u32::<LittleEndian>(TensorType::F32 as u32).unwrap();
                buf.write_u64::<LittleEndian>(0).unwrap();
            }

            while !buf.len().is_multiple_of(32) {
                buf.write_u8(0).unwrap();
            }

            for (_name, _shape, data) in &self.tensors {
                buf.extend_from_slice(data);
            }

            fs::write(path, &buf).unwrap();
        }
    }

    #[test]
    fn layer_forward_runs_without_panic() {
        let tmp = std::env::temp_dir().join("inferox_layer_forward_test.gguf");

        let hidden_size = 8usize;
        let intermediate_size = 16usize;
        let n_layers = 1usize;

        let mut builder = TestGgufBuilder::new();
        builder
            .metadata_string("general.architecture", "llama")
            .metadata_u32("llama.vocab_size", 100)
            .metadata_u32("llama.embedding_length", hidden_size as u32)
            .metadata_u32("llama.block_count", n_layers as u32)
            .metadata_u32("llama.attention.head_count", 2)
            .metadata_u32("llama.attention.head_count_kv", 2)
            .metadata_u32("llama.feed_forward_length", intermediate_size as u32)
            .metadata_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
            .metadata_u32("llama.context_length", 128);

        // One layer of weights
        let ones_hs = vec![1.0f32; hidden_size];
        let zeros_hs = vec![0.0f32; hidden_size];
        let ones_hshs = vec![1.0f32; hidden_size * hidden_size];
        let ones_ishs = vec![1.0f32; intermediate_size * hidden_size];
        let ones_hsint = vec![1.0f32; hidden_size * intermediate_size];

        builder
            .tensor_f32("token_embd.weight", &[1, hidden_size as u64], &zeros_hs)
            .tensor_f32("output_norm.weight", &[hidden_size as u64], &ones_hs)
            .tensor_f32("output.weight", &[1, hidden_size as u64], &zeros_hs)
            .tensor_f32("blk.0.attn_norm.weight", &[hidden_size as u64], &ones_hs)
            .tensor_f32("blk.0.attn_q.weight", &[hidden_size as u64, hidden_size as u64], &ones_hshs)
            .tensor_f32("blk.0.attn_k.weight", &[hidden_size as u64, hidden_size as u64], &ones_hshs)
            .tensor_f32("blk.0.attn_v.weight", &[hidden_size as u64, hidden_size as u64], &ones_hshs)
            .tensor_f32("blk.0.attn_output.weight", &[hidden_size as u64, hidden_size as u64], &ones_hshs)
            .tensor_f32("blk.0.ffn_norm.weight", &[hidden_size as u64], &ones_hs)
            .tensor_f32("blk.0.ffn_gate.weight", &[intermediate_size as u64, hidden_size as u64], &ones_ishs)
            .tensor_f32("blk.0.ffn_up.weight", &[intermediate_size as u64, hidden_size as u64], &ones_ishs)
            .tensor_f32("blk.0.ffn_down.weight", &[hidden_size as u64, intermediate_size as u64], &ones_hsint);

        builder.write(&tmp);

        let model = Model::open(&tmp).unwrap();
        let mut x = vec![0.1f32; hidden_size];
        let mut buf = LayerBuffers::new(1, &model.config);

        model.layers[0].forward(&model, &mut x, &mut buf).unwrap();

        assert_eq!(x.len(), hidden_size);

        fs::remove_file(&tmp).unwrap();
    }
}
