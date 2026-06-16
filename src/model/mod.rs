//! Model architecture loaded from a GGUF file.
//!
//! This module parses standard `llama.*` metadata keys and wires the tensor
//! info table into a structured `Model` with a `Config` and per-layer weight
//! indices. It does not yet run a forward pass; it only prepares the graph.

use crate::gguf::{GgufFile, Weights};
use std::path::Path;

pub use self::config::Config;

mod config;

/// A single transformer layer's weight tensor indices.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub index: usize,
    pub attn_norm: usize,
    pub attn_q: usize,
    pub attn_k: usize,
    pub attn_v: usize,
    pub attn_output: usize,
    pub ffn_norm: usize,
    pub ffn_gate: usize,
    pub ffn_up: usize,
    pub ffn_down: usize,
}

impl Layer {
    fn from_weights(weights: &GgufFile, index: usize) -> Option<Self> {
        let prefix = format!("blk.{index}.");
        Some(Self {
            index,
            attn_norm: weights.tensor_index(&format!("{prefix}attn_norm.weight"))?,
            attn_q: weights.tensor_index(&format!("{prefix}attn_q.weight"))?,
            attn_k: weights.tensor_index(&format!("{prefix}attn_k.weight"))?,
            attn_v: weights.tensor_index(&format!("{prefix}attn_v.weight"))?,
            attn_output: weights.tensor_index(&format!("{prefix}attn_output.weight"))?,
            ffn_norm: weights.tensor_index(&format!("{prefix}ffn_norm.weight"))?,
            ffn_gate: weights.tensor_index(&format!("{prefix}ffn_gate.weight"))?,
            ffn_up: weights.tensor_index(&format!("{prefix}ffn_up.weight"))?,
            ffn_down: weights.tensor_index(&format!("{prefix}ffn_down.weight"))?,
        })
    }
}

/// A loaded model with its configuration and wired tensor indices.
#[derive(Debug)]
pub struct Model {
    pub config: Config,
    pub weights: Weights,
    pub layers: Vec<Layer>,
    pub token_embeddings: usize,
    pub output_norm: usize,
    pub output: Option<usize>,
}

impl Model {
    /// Open a GGUF file, parse its architecture metadata, and wire tensors.
    pub fn open<P: AsRef<Path>>(path: P) -> crate::gguf::Result<Self> {
        let weights = Weights::open(path)?;
        let file = weights.file();
        let config = Config::from_gguf(file)?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let layer = Layer::from_weights(file, i)
                .ok_or_else(|| crate::gguf::Error::TensorNotFound(format!("blk.{i}.*")))?;
            layers.push(layer);
        }

        let token_embeddings = file
            .tensor_index("token_embd.weight")
            .ok_or_else(|| crate::gguf::Error::TensorNotFound("token_embd.weight".to_string()))?;
        let output_norm = file
            .tensor_index("output_norm.weight")
            .ok_or_else(|| crate::gguf::Error::TensorNotFound("output_norm.weight".to_string()))?;
        let output = file.tensor_index("output.weight");

        Ok(Self {
            config,
            weights,
            layers,
            token_embeddings,
            output_norm,
            output,
        })
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

    fn write_metadata_string(buf: &mut Vec<u8>, key: &str, value: &str) {
        write_string(buf, key);
        buf.write_u32::<LittleEndian>(ValueType::String as u32).unwrap();
        write_string(buf, value);
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

    fn write_tensor_info(buf: &mut Vec<u8>, name: &str, shape: &[u64]) {
        write_string(buf, name);
        buf.write_u32::<LittleEndian>(shape.len() as u32).unwrap();
        for &dim in shape {
            buf.write_u64::<LittleEndian>(dim).unwrap();
        }
        buf.write_u32::<LittleEndian>(TensorType::F32 as u32).unwrap();
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset ignored by wiring
    }

    fn write_test_llama_gguf(path: &Path) {
        let mut buf: Vec<u8> = Vec::new();

        // Header
        buf.write_all(b"GGUF").unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap();

        let tensor_names: Vec<String> = {
            let mut names = vec![
                "token_embd.weight".to_string(),
                "output_norm.weight".to_string(),
                "output.weight".to_string(),
            ];
            for i in 0..2 {
                names.extend([
                    format!("blk.{i}.attn_norm.weight"),
                    format!("blk.{i}.attn_q.weight"),
                    format!("blk.{i}.attn_k.weight"),
                    format!("blk.{i}.attn_v.weight"),
                    format!("blk.{i}.attn_output.weight"),
                    format!("blk.{i}.ffn_norm.weight"),
                    format!("blk.{i}.ffn_gate.weight"),
                    format!("blk.{i}.ffn_up.weight"),
                    format!("blk.{i}.ffn_down.weight"),
                ]);
            }
            names
        };

        buf.write_u64::<LittleEndian>(tensor_names.len() as u64).unwrap(); // n_tensors
        buf.write_u64::<LittleEndian>(9).unwrap(); // n_kv

        // Metadata
        write_metadata_string(&mut buf, "general.architecture", "llama");
        write_metadata_u32(&mut buf, "llama.vocab_size", 100);
        write_metadata_u32(&mut buf, "llama.embedding_length", 64);
        write_metadata_u32(&mut buf, "llama.block_count", 2);
        write_metadata_u32(&mut buf, "llama.attention.head_count", 4);
        write_metadata_u32(&mut buf, "llama.attention.head_count_kv", 2);
        write_metadata_u32(&mut buf, "llama.feed_forward_length", 128);
        write_metadata_f32(&mut buf, "llama.attention.layer_norm_rms_epsilon", 1e-5);
        write_metadata_u32(&mut buf, "llama.context_length", 512);

        // Tensor info table
        for name in &tensor_names {
            write_tensor_info(&mut buf, name, &[1, 1]);
        }

        // Padding to 32-byte alignment
        while !buf.len().is_multiple_of(32) {
            buf.write_u8(0).unwrap();
        }

        // Minimal tensor data
        for _ in 0..tensor_names.len() * 4 {
            buf.write_f32::<LittleEndian>(0.0).unwrap();
        }

        fs::write(path, &buf).unwrap();
    }

    #[test]
    fn config_parsed_from_metadata() {
        let tmp = std::env::temp_dir().join("inferox_model_config_test.gguf");
        write_test_llama_gguf(&tmp);

        let weights = Weights::open(&tmp).unwrap();
        let config = Config::from_gguf(weights.file()).unwrap();

        assert_eq!(config.architecture, "llama");
        assert_eq!(config.vocab_size, 100);
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.n_layers, 2);
        assert_eq!(config.n_heads, 4);
        assert_eq!(config.n_kv_heads, 2);
        assert_eq!(config.head_dim, 16); // 64 / 4
        assert_eq!(config.intermediate_size, 128);
        assert!((config.norm_eps - 1e-5).abs() < 1e-10);
        assert_eq!(config.context_length, 512);

        fs::remove_file(&tmp).unwrap();
    }

    #[test]
    fn model_wires_layers_and_embeddings() {
        let tmp = std::env::temp_dir().join("inferox_model_wiring_test.gguf");
        write_test_llama_gguf(&tmp);

        let model = Model::open(&tmp).unwrap();

        assert_eq!(model.config.n_layers, 2);
        assert_eq!(model.layers.len(), 2);
        assert_eq!(model.weights.file().tensors[model.token_embeddings].name, "token_embd.weight");
        assert_eq!(model.weights.file().tensors[model.output_norm].name, "output_norm.weight");
        assert_eq!(model.output, Some(2)); // index of output.weight

        let layer0 = &model.layers[0];
        assert_eq!(layer0.index, 0);
        assert_eq!(model.weights.file().tensors[layer0.attn_q].name, "blk.0.attn_q.weight");
        assert_eq!(model.weights.file().tensors[layer0.ffn_down].name, "blk.0.ffn_down.weight");

        fs::remove_file(&tmp).unwrap();
    }
}
