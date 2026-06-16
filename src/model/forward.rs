//! Transformer layer forward pass.

use crate::gguf::Result;
use crate::model::{Config, Layer, Model};
use crate::model::cache::{DecodeBuffers, KvCache};
use crate::ops::{
    add_in_place, embedding_lookup, matmul_t, rms_norm, rope, softmax_rows, swiglu,
};

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

    /// Run one transformer layer decode step in-place on a single token.
    ///
    /// Operates on `buf.hidden`, which has shape `[hidden_size]`. The KV-cache
    /// for this layer is updated with the new token's key and value before
    /// attention is computed.
    pub fn decode_step(
        &self,
        model: &Model,
        kv_cache: &mut [f32],
        v_cache: &mut [f32],
        cache_len: usize,
        buf: &mut DecodeBuffers,
    ) -> Result<()> {
        let cfg = &model.config;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let kv_group_size = cfg.n_heads / cfg.n_kv_heads;

        // --- Attention branch ---
        buf.residual.copy_from_slice(&buf.hidden);

        let attn_norm = model.weights.tensor_f32_by_index(self.attn_norm)?;
        rms_norm(&mut buf.hidden, &attn_norm, cfg.norm_eps);

        let w_q = model.weights.tensor_f32_by_index(self.attn_q)?;
        let w_k = model.weights.tensor_f32_by_index(self.attn_k)?;
        let w_v = model.weights.tensor_f32_by_index(self.attn_v)?;

        matmul_t(&buf.hidden, &w_q, &mut buf.q, 1, cfg.hidden_size, cfg.hidden_size);
        matmul_t(&buf.hidden, &w_k, &mut buf.k, 1, kv_dim, cfg.hidden_size);
        matmul_t(&buf.hidden, &w_v, &mut buf.v, 1, kv_dim, cfg.hidden_size);

        // RoPE at the current cache position
        rope(&mut buf.q, 1, cfg.n_heads, cfg.head_dim, cfg.rope_dim, cfg.rope_base);
        rope(&mut buf.k, 1, cfg.n_kv_heads, cfg.head_dim, cfg.rope_dim, cfg.rope_base);

        // Append k and v to the cache
        let kv_offset = cache_len * kv_dim;
        kv_cache[kv_offset..kv_offset + kv_dim].copy_from_slice(&buf.k);
        v_cache[kv_offset..kv_offset + kv_dim].copy_from_slice(&buf.v);
        let new_cache_len = cache_len + 1;

        // Multi-head attention with GQA over the cached sequence
        buf.attn_out.fill(0.0);

        for h_q in 0..cfg.n_heads {
            let h_kv = h_q / kv_group_size;
            let q_offset = h_q * cfg.head_dim;
            let q_head = &buf.q[q_offset..q_offset + cfg.head_dim];

            for t in 0..new_cache_len {
                let k_offset = t * kv_dim + h_kv * cfg.head_dim;
                let k_head = &kv_cache[k_offset..k_offset + cfg.head_dim];

                let mut score = 0.0f32;
                for (&q_val, &k_val) in q_head.iter().zip(k_head) {
                    score += q_val * k_val;
                }
                buf.scores[t] = score / (cfg.head_dim as f32).sqrt();
            }

            softmax_rows(&mut buf.scores[..new_cache_len], 1, new_cache_len);

            buf.head_out[..cfg.head_dim].fill(0.0);
            for t in 0..new_cache_len {
                let score = buf.scores[t];
                let v_offset = t * kv_dim + h_kv * cfg.head_dim;
                let v_head = &v_cache[v_offset..v_offset + cfg.head_dim];
                for (d, &v_val) in v_head.iter().enumerate() {
                    buf.head_out[d] += score * v_val;
                }
            }

            let attn_offset = h_q * cfg.head_dim;
            buf.attn_out[attn_offset..attn_offset + cfg.head_dim]
                .copy_from_slice(&buf.head_out[..cfg.head_dim]);
        }

        // Output projection
        let w_o = model.weights.tensor_f32_by_index(self.attn_output)?;
        matmul_t(&buf.attn_out, &w_o, &mut buf.hidden, 1, cfg.hidden_size, cfg.hidden_size);
        add_in_place(&mut buf.hidden, &buf.residual);

        // --- FFN branch ---
        buf.residual.copy_from_slice(&buf.hidden);

        let ffn_norm = model.weights.tensor_f32_by_index(self.ffn_norm)?;
        rms_norm(&mut buf.hidden, &ffn_norm, cfg.norm_eps);

        let w_gate = model.weights.tensor_f32_by_index(self.ffn_gate)?;
        let w_up = model.weights.tensor_f32_by_index(self.ffn_up)?;
        let w_down = model.weights.tensor_f32_by_index(self.ffn_down)?;

        matmul_t(&buf.hidden, &w_gate, &mut buf.ffn_gate, 1, cfg.intermediate_size, cfg.hidden_size);
        matmul_t(&buf.hidden, &w_up, &mut buf.ffn_up, 1, cfg.intermediate_size, cfg.hidden_size);
        swiglu(&buf.ffn_gate, &buf.ffn_up, &mut buf.ffn_mid);
        matmul_t(&buf.ffn_mid, &w_down, &mut buf.ffn_down, 1, cfg.hidden_size, cfg.intermediate_size);

        buf.hidden.copy_from_slice(&buf.ffn_down);
        add_in_place(&mut buf.hidden, &buf.residual);

        Ok(())
    }
}

impl Model {
    /// Run a full forward pass on a prompt and return logits for every token.
    ///
    /// Output shape: `[input_ids.len(), vocab_size]`. For generation you
    /// typically only need the last row.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let seq_len = input_ids.len();
        let vocab_size = self.config.vocab_size as usize;
        let hidden_size = self.config.hidden_size;

        if input_ids.iter().any(|&id| id as usize >= vocab_size) {
            return Err(crate::gguf::Error::InvalidTensorIndex(0));
        }

        let embeddings = self.weights.tensor_f32_by_index(self.token_embeddings)?;
        let mut hidden = vec![0.0f32; seq_len * hidden_size];
        embedding_lookup(&embeddings, input_ids, hidden_size, &mut hidden);

        let mut buf = LayerBuffers::new(seq_len, &self.config);
        for layer in &self.layers {
            layer.forward(self, &mut hidden, &mut buf)?;
        }

        let output_norm = self.weights.tensor_f32_by_index(self.output_norm)?;
        rms_norm(&mut hidden, &output_norm, self.config.norm_eps);

        let mut logits = vec![0.0f32; seq_len * vocab_size];
        let output_idx = self.output.unwrap_or(self.token_embeddings);
        let output_weight = self.weights.tensor_f32_by_index(output_idx)?;
        matmul_t(
            &hidden,
            &output_weight,
            &mut logits,
            seq_len,
            vocab_size,
            hidden_size,
        );

        Ok(logits)
    }

    /// Convenience: forward a prompt and return logits for the last token only.
    pub fn forward_last_token_logits(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        let vocab_size = self.config.vocab_size as usize;
        let all_logits = self.forward(input_ids)?;
        let start = all_logits.len() - vocab_size;
        Ok(all_logits[start..].to_vec())
    }

    /// Run one autoregressive decode step for a single token id.
    ///
    /// Updates `kv_cache` and returns a slice to the logits for the next token.
    pub fn decode_step<'a>(
        &self,
        token_id: u32,
        kv_cache: &mut KvCache,
        buf: &'a mut DecodeBuffers,
    ) -> Result<&'a [f32]> {
        let hidden_size = self.config.hidden_size;
        let vocab_size = self.config.vocab_size as usize;

        if token_id as usize >= vocab_size {
            return Err(crate::gguf::Error::InvalidTensorIndex(token_id as usize));
        }

        let embeddings = self.weights.tensor_f32_by_index(self.token_embeddings)?;
        let src = token_id as usize * hidden_size;
        buf.hidden.copy_from_slice(&embeddings[src..src + hidden_size]);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            layer.decode_step(
                self,
                &mut kv_cache.k[layer_idx],
                &mut kv_cache.v[layer_idx],
                kv_cache.len,
                buf,
            )?;
        }

        let output_norm = self.weights.tensor_f32_by_index(self.output_norm)?;
        rms_norm(&mut buf.hidden, &output_norm, self.config.norm_eps);

        let output_idx = self.output.unwrap_or(self.token_embeddings);
        let output_weight = self.weights.tensor_f32_by_index(output_idx)?;
        matmul_t(
            &buf.hidden,
            &output_weight,
            &mut buf.logits,
            1,
            vocab_size,
            hidden_size,
        );

        kv_cache.len += 1;
        Ok(&buf.logits)
    }

    /// Greedy autoregressive generation from a prompt.
    ///
    /// Returns the generated token ids. `kv_cache` and `buf` must be sized for
    /// the model and a large enough context length.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        kv_cache: &mut KvCache,
        buf: &mut DecodeBuffers,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::new();
        kv_cache.clear();

        for &token in prompt_tokens {
            self.decode_step(token, kv_cache, buf)?;
        }

        for _ in 0..max_tokens {
            let next_token = argmax(&buf.logits);
            generated.push(next_token);
            self.decode_step(next_token, kv_cache, buf)?;
        }

        Ok(generated)
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
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
        let zeros_vocab_hs = vec![0.0f32; 100 * hidden_size];
        let ones_hshs = vec![1.0f32; hidden_size * hidden_size];
        let ones_ishs = vec![1.0f32; intermediate_size * hidden_size];
        let ones_hsint = vec![1.0f32; hidden_size * intermediate_size];

        builder
            .tensor_f32("token_embd.weight", &[100, hidden_size as u64], &zeros_vocab_hs)
            .tensor_f32("output_norm.weight", &[hidden_size as u64], &ones_hs)
            .tensor_f32("output.weight", &[100, hidden_size as u64], &zeros_vocab_hs)
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

        let logits = model.forward(&[0]).unwrap();
        assert_eq!(logits.len(), 100); // vocab_size

        fs::remove_file(&tmp).unwrap();
    }

    #[test]
    fn generate_with_kv_cache_runs() {
        let tmp = std::env::temp_dir().join("inferox_generate_test.gguf");

        let hidden_size = 8usize;
        let intermediate_size = 16usize;
        let n_layers = 1usize;
        let max_seq_len = 32usize;

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
            .metadata_u32("llama.context_length", max_seq_len as u32);

        let zeros_vocab_hs = vec![0.0f32; 100 * hidden_size];
        let zeros_hshs = vec![0.0f32; hidden_size * hidden_size];
        let zeros_ishs = vec![0.0f32; intermediate_size * hidden_size];
        let zeros_hsint = vec![0.0f32; hidden_size * intermediate_size];
        let ones_hs = vec![1.0f32; hidden_size];

        builder
            .tensor_f32("token_embd.weight", &[100, hidden_size as u64], &zeros_vocab_hs)
            .tensor_f32("output_norm.weight", &[hidden_size as u64], &ones_hs)
            .tensor_f32("output.weight", &[100, hidden_size as u64], &zeros_vocab_hs)
            .tensor_f32("blk.0.attn_norm.weight", &[hidden_size as u64], &ones_hs)
            .tensor_f32("blk.0.attn_q.weight", &[hidden_size as u64, hidden_size as u64], &zeros_hshs)
            .tensor_f32("blk.0.attn_k.weight", &[hidden_size as u64, hidden_size as u64], &zeros_hshs)
            .tensor_f32("blk.0.attn_v.weight", &[hidden_size as u64, hidden_size as u64], &zeros_hshs)
            .tensor_f32("blk.0.attn_output.weight", &[hidden_size as u64, hidden_size as u64], &zeros_hshs)
            .tensor_f32("blk.0.ffn_norm.weight", &[hidden_size as u64], &ones_hs)
            .tensor_f32("blk.0.ffn_gate.weight", &[intermediate_size as u64, hidden_size as u64], &zeros_ishs)
            .tensor_f32("blk.0.ffn_up.weight", &[intermediate_size as u64, hidden_size as u64], &zeros_ishs)
            .tensor_f32("blk.0.ffn_down.weight", &[hidden_size as u64, intermediate_size as u64], &zeros_hsint);

        builder.write(&tmp);

        let model = Model::open(&tmp).unwrap();
        let mut kv_cache = KvCache::new(n_layers, max_seq_len, &model.config);
        let mut buf = DecodeBuffers::new(max_seq_len, &model.config);

        let generated = model.generate(&[0, 1, 2], 5, &mut kv_cache, &mut buf).unwrap();
        assert_eq!(generated.len(), 5);
        assert!(generated.iter().all(|&t| t < 100));
        assert_eq!(kv_cache.len(), 3 + 5);

        fs::remove_file(&tmp).unwrap();
    }
}
