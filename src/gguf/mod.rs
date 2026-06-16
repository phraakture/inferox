//! GGUF (Georgi Gerganov Universal Format) parser.
//!
//! This crate module reads the metadata and tensor-info tables from GGUF files
//! as used by llama.cpp. It intentionally does **not** load tensor weight data
//! into memory; instead it exposes offsets and layouts so callers can mmap or
//! stream weights as needed.

mod error;
mod file;
mod tensor;
mod types;
mod value;
mod weights;

pub use error::{Error, Result};
pub use file::{GgufFile, Header, Metadata};
pub use tensor::TensorInfo;
pub use types::{BlockLayout, TensorType, ValueType};
pub use value::Value;
pub use weights::Weights;
