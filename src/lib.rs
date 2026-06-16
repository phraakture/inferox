//! Inferox — a Rust-native LLM runtime inspired by llama.cpp.
//!
//! Currently the crate exposes a GGUF parser and reference CPU compute kernels.
//! Model execution layers will be added incrementally.

pub mod gguf;
pub mod model;
pub mod ops;
