use crate::gguf::error::Result;
use crate::gguf::types::ValueType;
use byteorder::{LittleEndian, ReadBytesExt};
use std::fmt;
use std::io::Read;

/// A single GGUF metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(ValueType, Vec<Value>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl Value {
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Uint8(_) => ValueType::Uint8,
            Value::Int8(_) => ValueType::Int8,
            Value::Uint16(_) => ValueType::Uint16,
            Value::Int16(_) => ValueType::Int16,
            Value::Uint32(_) => ValueType::Uint32,
            Value::Int32(_) => ValueType::Int32,
            Value::Float32(_) => ValueType::Float32,
            Value::Bool(_) => ValueType::Bool,
            Value::String(_) => ValueType::String,
            Value::Array(_, _) => ValueType::Array,
            Value::Uint64(_) => ValueType::Uint64,
            Value::Int64(_) => ValueType::Int64,
            Value::Float64(_) => ValueType::Float64,
        }
    }

    /// Read a value of the given type from a little-endian byte stream.
    pub fn read<R: Read>(reader: &mut R, ty: ValueType) -> Result<Self> {
        use ValueType::*;
        let value = match ty {
            Uint8 => Value::Uint8(reader.read_u8()?),
            Int8 => Value::Int8(reader.read_i8()?),
            Uint16 => Value::Uint16(reader.read_u16::<LittleEndian>()?),
            Int16 => Value::Int16(reader.read_i16::<LittleEndian>()?),
            Uint32 => Value::Uint32(reader.read_u32::<LittleEndian>()?),
            Int32 => Value::Int32(reader.read_i32::<LittleEndian>()?),
            Float32 => Value::Float32(reader.read_f32::<LittleEndian>()?),
            Bool => Value::Bool(reader.read_u8()? != 0),
            String => Value::String(read_string(reader)?),
            Array => {
                let item_type = ValueType::from_u32(reader.read_u32::<LittleEndian>()?)?;
                let len = reader.read_u64::<LittleEndian>()?;
                let mut items = Vec::with_capacity(len.min(1 << 20) as usize);
                for _ in 0..len {
                    items.push(Value::read(reader, item_type)?);
                }
                Value::Array(item_type, items)
            }
            Uint64 => Value::Uint64(reader.read_u64::<LittleEndian>()?),
            Int64 => Value::Int64(reader.read_i64::<LittleEndian>()?),
            Float64 => Value::Float64(reader.read_f64::<LittleEndian>()?),
        };
        Ok(value)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Uint8(v) => write!(f, "{v}"),
            Value::Int8(v) => write!(f, "{v}"),
            Value::Uint16(v) => write!(f, "{v}"),
            Value::Int16(v) => write!(f, "{v}"),
            Value::Uint32(v) => write!(f, "{v}"),
            Value::Int32(v) => write!(f, "{v}"),
            Value::Float32(v) => write!(f, "{v}"),
            Value::Bool(v) => write!(f, "{v}"),
            Value::String(v) => write!(f, "{v}"),
            Value::Array(ty, items) => {
                write!(f, "{ty}[{}] [", items.len())?;
                let max_preview = 8;
                for (i, item) in items.iter().enumerate() {
                    if i >= max_preview && items.len() > max_preview + 1 {
                        write!(f, ", ...")?;
                        break;
                    }
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, "]")
            }
            Value::Uint64(v) => write!(f, "{v}"),
            Value::Int64(v) => write!(f, "{v}"),
            Value::Float64(v) => write!(f, "{v}"),
        }
    }
}

pub fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = reader.read_u64::<LittleEndian>()?;
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}
