//! Tokenizers for converting between text and token ids.
//!
//! Currently implements a minimal byte-level BPE tokenizer compatible with the
//! Hugging Face `tokenizer.json` format used by models like GPT-2, Qwen2, and
//! SmolLM. Pre-tokenizers (regex splitting) are not yet supported.

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

mod bytes;

pub use bytes::{byte_decoder, byte_encoder};

/// Anything that can turn text into token ids and back.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
}

/// A byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// token string -> id
    vocab: HashMap<String, u32>,
    /// id -> token string
    id_to_token: HashMap<u32, String>,
    /// (first, second) -> merge rank (lower rank = higher priority)
    merges: HashMap<(String, String), usize>,
    /// Special tokens that should be split out before BPE.
    special_tokens: HashSet<String>,
    /// Token id of the unknown token.
    unk_id: u32,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    model: Model,
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    content: String,
}

#[derive(Debug, Deserialize)]
struct Model {
    #[serde(rename = "type")]
    kind: String,
    vocab: HashMap<String, u32>,
    merges: Vec<String>,
    #[serde(rename = "unk_token")]
    unk_token: Option<String>,
}

impl BpeTokenizer {
    /// Load a BPE tokenizer from a Hugging Face `tokenizer.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, TokenizerError> {
        let raw = fs::read_to_string(path)?;
        Self::from_json(&raw)
    }

    /// Load a BPE tokenizer from a JSON string.
    pub fn from_json(raw: &str) -> Result<Self, TokenizerError> {
        let parsed: TokenizerJson = serde_json::from_str(raw)?;

        if parsed.model.kind != "BPE" {
            return Err(TokenizerError::UnsupportedModel(parsed.model.kind));
        }

        let mut id_to_token = HashMap::with_capacity(parsed.model.vocab.len());
        for (token, id) in &parsed.model.vocab {
            id_to_token.insert(*id, token.clone());
        }

        let mut merges = HashMap::new();
        for (rank, line) in parsed.model.merges.iter().enumerate() {
            let mut parts = line.split_whitespace();
            let a = parts.next().unwrap_or("").to_string();
            let b = parts.next().unwrap_or("").to_string();
            if !a.is_empty() && !b.is_empty() {
                merges.insert((a, b), rank);
            }
        }

        let special_tokens: HashSet<String> = parsed
            .added_tokens
            .into_iter()
            .map(|t| t.content)
            .collect();

        let unk_id = parsed
            .model
            .unk_token
            .and_then(|t| parsed.model.vocab.get(&t).copied())
            .unwrap_or(0);

        Ok(Self {
            vocab: parsed.model.vocab,
            id_to_token,
            merges,
            special_tokens,
            unk_id,
        })
    }

    fn bpe(&self, mut pieces: Vec<String>) -> Vec<String> {
        if pieces.len() < 2 {
            return pieces;
        }

        loop {
            let mut best_rank: Option<usize> = None;
            let mut best_idx: usize = 0;

            for i in 0..pieces.len() - 1 {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merges.get(&pair)
                    && (best_rank.is_none() || rank < best_rank.unwrap())
                {
                    best_rank = Some(rank);
                    best_idx = i;
                }
            }

            if let Some(idx) = best_rank.map(|_| best_idx) {
                let merged = format!("{}{}", pieces[idx], pieces[idx + 1]);
                pieces[idx] = merged;
                pieces.remove(idx + 1);
            } else {
                break;
            }
        }

        pieces
    }

    fn encode_piece(&self, text: &str) -> Vec<u32> {
        let encoder = byte_encoder();
        let bytes: Vec<u8> = text.bytes().collect();
        let initial: Vec<String> = bytes
            .iter()
            .map(|&b| encoder[b as usize].to_string())
            .collect();

        let merged = self.bpe(initial);
        merged
            .into_iter()
            .map(|s| *self.vocab.get(&s).unwrap_or(&self.unk_id))
            .collect()
    }

    fn split_special(&self, text: &str) -> Vec<String> {
        if self.special_tokens.is_empty() {
            return vec![text.to_string()];
        }

        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut found: Option<(usize, &str)> = None;
            for token in &self.special_tokens {
                if let Some(pos) = remaining.find(token)
                    && (found.is_none() || pos < found.unwrap().0)
                {
                    found = Some((pos, token));
                }
            }

            match found {
                Some((0, token)) => {
                    result.push(token.to_string());
                    remaining = &remaining[token.len()..];
                }
                Some((pos, token)) => {
                    result.push(remaining[..pos].to_string());
                    result.push(token.to_string());
                    remaining = &remaining[pos + token.len()..];
                }
                None => {
                    result.push(remaining.to_string());
                    break;
                }
            }
        }

        result
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for piece in self.split_special(text) {
            if self.special_tokens.contains(&piece) {
                ids.push(*self.vocab.get(&piece).unwrap_or(&self.unk_id));
            } else {
                ids.extend(self.encode_piece(&piece));
            }
        }
        ids
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let decoder = byte_decoder();
        let mut bytes = Vec::new();

        for &id in tokens {
            if let Some(token) = self.id_to_token.get(&id) {
                for ch in token.chars() {
                    if let Some(&b) = decoder.get(&ch) {
                        bytes.push(b);
                    } else {
                        // Non-byte token; encode its UTF-8 bytes directly.
                        bytes.extend(ch.encode_utf8(&mut [0; 4]).as_bytes());
                    }
                }
            }
        }

        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    UnsupportedModel(String),
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::Io(e) => write!(f, "io error: {e}"),
            TokenizerError::Json(e) => write!(f, "json error: {e}"),
            TokenizerError::UnsupportedModel(t) => write!(f, "unsupported tokenizer model: {t}"),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::Io(e) => Some(e),
            TokenizerError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TokenizerError {
    fn from(e: std::io::Error) -> Self {
        TokenizerError::Io(e)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(e: serde_json::Error) -> Self {
        TokenizerError::Json(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tokenizer_json() -> String {
        let encoder = byte_encoder();
        let mut vocab = HashMap::new();
        vocab.insert("<|endoftext|>".to_string(), 0);

        // Add single-byte tokens.
        for b in 0u8..=255u8 {
            let mut s = String::new();
            s.push(encoder[b as usize]);
            vocab.insert(s, b as u32 + 1);
        }

        // Add a merge: space+space -> "  " (using byte 32's encoder char).
        let space = encoder[32].to_string();
        let mut merges = Vec::new();
        merges.push(format!("{} {}", space, space));

        serde_json::json!({
            "version": "1.0",
            "added_tokens": [{"content": "<|endoftext|>"}],
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": merges,
                "unk_token": "<|endoftext|>"
            }
        })
        .to_string()
    }

    #[test]
    fn encode_decode_ascii_roundtrip() {
        let tokenizer = BpeTokenizer::from_json(&tiny_tokenizer_json()).unwrap();
        let text = "hello world";
        let ids = tokenizer.encode(text);
        assert!(!ids.is_empty());
        let decoded = tokenizer.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn special_token_is_split() {
        let tokenizer = BpeTokenizer::from_json(&tiny_tokenizer_json()).unwrap();
        let ids = tokenizer.encode("hi<|endoftext|>lo");
        assert!(ids.contains(&0));
    }
}
