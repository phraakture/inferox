use crate::gguf::error::{Error, Result};
use crate::gguf::tensor::TensorInfo;
use crate::gguf::types::ValueType;
use crate::gguf::value::{read_string, Value};
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

/// Parsed header of a GGUF file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
}

/// A metadata key-value pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub key: String,
    pub value: Value,
}

impl Metadata {
    pub fn value_type(&self) -> ValueType {
        self.value.value_type()
    }
}

/// A fully parsed GGUF file descriptor.
///
/// This structure does **not** own the tensor weight data; it only holds the
/// metadata and tensor info tables. Use [`TensorInfo::data_offset`] with
/// [`GgufFile::tensor_data_offset`] to locate raw tensor bytes in the file.
#[derive(Clone, Debug, PartialEq)]
pub struct GgufFile {
    pub header: Header,
    pub metadata: Vec<Metadata>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where the tensor data region begins.
    pub tensor_data_offset: u64,
}

impl GgufFile {
    /// Parse a GGUF file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Self::read(reader)
    }

    /// Parse a GGUF file from any seekable reader.
    pub fn read<R: Read + Seek>(mut reader: R) -> Result<Self> {
        let header = Self::read_header(&mut reader)?;
        let metadata = Self::read_metadata(&mut reader, header.n_kv)?;
        let tensors = Self::read_tensors(&mut reader, header.n_tensors)?;
        let tensor_data_offset = Self::align_offset(reader.stream_position()?);

        Ok(Self {
            header,
            metadata,
            tensors,
            tensor_data_offset,
        })
    }

    fn read_header<R: Read>(reader: &mut R) -> Result<Header> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != GGUF_MAGIC {
            return Err(Error::InvalidMagic(magic));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(Error::UnsupportedVersion(version));
        }

        let n_tensors = reader.read_u64::<LittleEndian>()?;
        let n_kv = reader.read_u64::<LittleEndian>()?;

        Ok(Header {
            version,
            n_tensors,
            n_kv,
        })
    }

    fn read_metadata<R: Read>(reader: &mut R, n_kv: u64) -> Result<Vec<Metadata>> {
        let mut metadata = Vec::with_capacity(n_kv.min(1 << 20) as usize);
        for _ in 0..n_kv {
            let key = read_string(reader)?;
            let value_type = ValueType::from_u32(reader.read_u32::<LittleEndian>()?)?;
            let value = Value::read(reader, value_type)?;
            metadata.push(Metadata { key, value });
        }
        Ok(metadata)
    }

    fn read_tensors<R: Read>(reader: &mut R, n_tensors: u64) -> Result<Vec<TensorInfo>> {
        let mut tensors = Vec::with_capacity(n_tensors.min(1 << 20) as usize);
        for _ in 0..n_tensors {
            tensors.push(TensorInfo::read(reader)?);
        }
        Ok(tensors)
    }

    /// GGUF aligns the tensor data region to 32 bytes.
    fn align_offset(offset: u64) -> u64 {
        const ALIGNMENT: u64 = 32;
        offset.div_ceil(ALIGNMENT) * ALIGNMENT
    }

    /// Look up a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.metadata.iter().find(|m| m.key == key).map(|m| &m.value)
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Find the index of a tensor by name.
    pub fn tensor_index(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    /// Total byte size of all tensor weight data, if every type is known.
    pub fn total_tensor_data_size(&self) -> Option<u64> {
        self.tensors
            .iter()
            .map(|t| t.byte_size().map(|s| s as u64))
            .sum()
    }

    /// Get a metadata value as `u32`.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.get(key)? {
            crate::gguf::value::Value::Uint8(v) => Some(*v as u32),
            crate::gguf::value::Value::Int8(v) => Some(*v as u32),
            crate::gguf::value::Value::Uint16(v) => Some(*v as u32),
            crate::gguf::value::Value::Int16(v) => Some(*v as u32),
            crate::gguf::value::Value::Uint32(v) => Some(*v),
            crate::gguf::value::Value::Int32(v) => Some(*v as u32),
            crate::gguf::value::Value::Uint64(v) => Some(*v as u32),
            crate::gguf::value::Value::Int64(v) => Some(*v as u32),
            _ => None,
        }
    }

    /// Get a metadata value as `u64`.
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.get(key)? {
            crate::gguf::value::Value::Uint8(v) => Some(*v as u64),
            crate::gguf::value::Value::Int8(v) => Some(*v as u64),
            crate::gguf::value::Value::Uint16(v) => Some(*v as u64),
            crate::gguf::value::Value::Int16(v) => Some(*v as u64),
            crate::gguf::value::Value::Uint32(v) => Some(*v as u64),
            crate::gguf::value::Value::Int32(v) => Some(*v as u64),
            crate::gguf::value::Value::Uint64(v) => Some(*v),
            crate::gguf::value::Value::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Get a metadata value as `f32`.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.get(key)? {
            crate::gguf::value::Value::Float32(v) => Some(*v),
            crate::gguf::value::Value::Float64(v) => Some(*v as f32),
            crate::gguf::value::Value::Uint8(v) => Some(*v as f32),
            crate::gguf::value::Value::Int8(v) => Some(*v as f32),
            crate::gguf::value::Value::Uint16(v) => Some(*v as f32),
            crate::gguf::value::Value::Int16(v) => Some(*v as f32),
            crate::gguf::value::Value::Uint32(v) => Some(*v as f32),
            crate::gguf::value::Value::Int32(v) => Some(*v as f32),
            crate::gguf::value::Value::Uint64(v) => Some(*v as f32),
            crate::gguf::value::Value::Int64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Get a metadata value as a string slice.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            crate::gguf::value::Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}
