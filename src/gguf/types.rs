use crate::gguf::error::{Error, Result};
use std::fmt;

/// GGUF metadata value types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl ValueType {
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(ValueType::Uint8),
            1 => Ok(ValueType::Int8),
            2 => Ok(ValueType::Uint16),
            3 => Ok(ValueType::Int16),
            4 => Ok(ValueType::Uint32),
            5 => Ok(ValueType::Int32),
            6 => Ok(ValueType::Float32),
            7 => Ok(ValueType::Bool),
            8 => Ok(ValueType::String),
            9 => Ok(ValueType::Array),
            10 => Ok(ValueType::Uint64),
            11 => Ok(ValueType::Int64),
            12 => Ok(ValueType::Float64),
            _ => Err(Error::UnknownValueType(value)),
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ValueType::Uint8 => "uint8",
            ValueType::Int8 => "int8",
            ValueType::Uint16 => "uint16",
            ValueType::Int16 => "int16",
            ValueType::Uint32 => "uint32",
            ValueType::Int32 => "int32",
            ValueType::Float32 => "float32",
            ValueType::Bool => "bool",
            ValueType::String => "string",
            ValueType::Array => "array",
            ValueType::Uint64 => "uint64",
            ValueType::Int64 => "int64",
            ValueType::Float64 => "float64",
        };
        f.write_str(s)
    }
}

/// GGML tensor data types stored inside a GGUF file.
///
/// Quantized types are stored in fixed-size blocks; see [`TensorType::block_layout`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 5,
    Q5_1 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    I64 = 19,
    F64 = 20,
    Iq1S = 21,
    Iq1M = 22,
    Iq2Xxs = 23,
    Iq2Xs = 24,
    Iq2S = 25,
    Iq3Xxs = 26,
    Iq3S = 27,
    Iq4Xs = 28,
    Iq4Nl = 29,
    Iq4Xss = 30,
}

/// Number of elements per block and bytes per block for a quantized tensor type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockLayout {
    pub elements: usize,
    pub bytes: usize,
}

impl TensorType {
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(TensorType::F32),
            1 => Ok(TensorType::F16),
            2 => Ok(TensorType::Q4_0),
            3 => Ok(TensorType::Q4_1),
            5 => Ok(TensorType::Q5_0),
            6 => Ok(TensorType::Q5_1),
            7 => Ok(TensorType::Q8_0),
            8 => Ok(TensorType::Q8_1),
            10 => Ok(TensorType::Q2K),
            11 => Ok(TensorType::Q3K),
            12 => Ok(TensorType::Q4K),
            13 => Ok(TensorType::Q5K),
            14 => Ok(TensorType::Q6K),
            15 => Ok(TensorType::Q8K),
            16 => Ok(TensorType::I8),
            17 => Ok(TensorType::I16),
            18 => Ok(TensorType::I32),
            19 => Ok(TensorType::I64),
            20 => Ok(TensorType::F64),
            21 => Ok(TensorType::Iq1S),
            22 => Ok(TensorType::Iq1M),
            23 => Ok(TensorType::Iq2Xxs),
            24 => Ok(TensorType::Iq2Xs),
            25 => Ok(TensorType::Iq2S),
            26 => Ok(TensorType::Iq3Xxs),
            27 => Ok(TensorType::Iq3S),
            28 => Ok(TensorType::Iq4Xs),
            29 => Ok(TensorType::Iq4Nl),
            30 => Ok(TensorType::Iq4Xss),
            _ => Err(Error::UnknownTensorType(value)),
        }
    }

    /// Human-readable name, matching llama.cpp conventions.
    pub fn name(self) -> &'static str {
        match self {
            TensorType::F32 => "F32",
            TensorType::F16 => "F16",
            TensorType::Q4_0 => "Q4_0",
            TensorType::Q4_1 => "Q4_1",
            TensorType::Q5_0 => "Q5_0",
            TensorType::Q5_1 => "Q5_1",
            TensorType::Q8_0 => "Q8_0",
            TensorType::Q8_1 => "Q8_1",
            TensorType::Q2K => "Q2_K",
            TensorType::Q3K => "Q3_K",
            TensorType::Q4K => "Q4_K",
            TensorType::Q5K => "Q5_K",
            TensorType::Q6K => "Q6_K",
            TensorType::Q8K => "Q8_K",
            TensorType::I8 => "I8",
            TensorType::I16 => "I16",
            TensorType::I32 => "I32",
            TensorType::I64 => "I64",
            TensorType::F64 => "F64",
            TensorType::Iq1S => "IQ1_S",
            TensorType::Iq1M => "IQ1_M",
            TensorType::Iq2Xxs => "IQ2_XXS",
            TensorType::Iq2Xs => "IQ2_XS",
            TensorType::Iq2S => "IQ2_S",
            TensorType::Iq3Xxs => "IQ3_XXS",
            TensorType::Iq3S => "IQ3_S",
            TensorType::Iq4Xs => "IQ4_XS",
            TensorType::Iq4Nl => "IQ4_NL",
            TensorType::Iq4Xss => "IQ4_XSS",
        }
    }

    /// Size of a single element in bytes for non-quantized types.
    pub fn element_size(self) -> Option<usize> {
        match self {
            TensorType::F32 => Some(4),
            TensorType::F16 => Some(2),
            TensorType::I8 => Some(1),
            TensorType::I16 => Some(2),
            TensorType::I32 => Some(4),
            TensorType::I64 => Some(8),
            TensorType::F64 => Some(8),
            _ => None,
        }
    }

    /// Block layout for quantized types.
    ///
    /// Returns `None` for non-quantized types.
    pub fn block_layout(self) -> Option<BlockLayout> {
        match self {
            // 32-element blocks.
            TensorType::Q4_0 => Some(BlockLayout { elements: 32, bytes: 18 }),
            TensorType::Q4_1 => Some(BlockLayout { elements: 32, bytes: 20 }),
            TensorType::Q5_0 => Some(BlockLayout { elements: 32, bytes: 22 }),
            TensorType::Q5_1 => Some(BlockLayout { elements: 32, bytes: 24 }),
            TensorType::Q8_0 => Some(BlockLayout { elements: 32, bytes: 34 }),
            TensorType::Q8_1 => Some(BlockLayout { elements: 32, bytes: 36 }),
            TensorType::Iq4Nl => Some(BlockLayout { elements: 32, bytes: 18 }),

            // 256-element K-quant blocks.
            TensorType::Q2K => Some(BlockLayout { elements: 256, bytes: 96 }),
            TensorType::Q3K => Some(BlockLayout { elements: 256, bytes: 110 }),
            TensorType::Q4K => Some(BlockLayout { elements: 256, bytes: 144 }),
            TensorType::Q5K => Some(BlockLayout { elements: 256, bytes: 176 }),
            TensorType::Q6K => Some(BlockLayout { elements: 256, bytes: 210 }),
            TensorType::Q8K => Some(BlockLayout { elements: 256, bytes: 292 }),

            // 256-element IQ blocks (sizes from llama.cpp/ggml).
            TensorType::Iq1S => Some(BlockLayout { elements: 256, bytes: 166 }),
            TensorType::Iq1M => Some(BlockLayout { elements: 256, bytes: 184 }),
            TensorType::Iq2Xxs => Some(BlockLayout { elements: 256, bytes: 66 }),
            TensorType::Iq2Xs => Some(BlockLayout { elements: 256, bytes: 104 }),
            TensorType::Iq2S => Some(BlockLayout { elements: 256, bytes: 176 }),
            TensorType::Iq3Xxs => Some(BlockLayout { elements: 256, bytes: 98 }),
            TensorType::Iq3S => Some(BlockLayout { elements: 256, bytes: 166 }),
            TensorType::Iq4Xs => Some(BlockLayout { elements: 256, bytes: 136 }),
            TensorType::Iq4Xss => Some(BlockLayout { elements: 256, bytes: 116 }),

            _ => None,
        }
    }

    /// Returns true for quantized tensor types.
    pub fn is_quantized(self) -> bool {
        self.element_size().is_none()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
