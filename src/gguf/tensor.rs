use crate::gguf::error::{Error, Result};
use crate::gguf::types::TensorType;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

/// Description of a single tensor stored in a GGUF file.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub ty: TensorType,
    /// Offset in bytes from the start of the tensor data region.
    pub offset: u64,
}

impl TensorInfo {
    /// Number of elements in the tensor.
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Number of dimensions.
    pub fn n_dims(&self) -> usize {
        self.shape.len()
    }

    /// Byte size of the tensor's raw weight data.
    ///
    /// Returns `None` if the type is unknown to this parser.
    pub fn byte_size(&self) -> Option<usize> {
        let n = self.n_elements();
        if let Some(element_size) = self.ty.element_size() {
            Some(n as usize * element_size)
        } else if let Some(layout) = self.ty.block_layout() {
            let blocks = n.div_ceil(layout.elements as u64);
            Some(blocks as usize * layout.bytes)
        } else {
            None
        }
    }

    /// Absolute file offset where the tensor's data begins.
    ///
    /// `tensor_data_offset` is the offset of the tensor data region from the
    /// beginning of the file (after the header, metadata, tensor info, and
    /// alignment padding).
    pub fn data_offset(&self, tensor_data_offset: u64) -> u64 {
        tensor_data_offset + self.offset
    }

    /// Read tensor info from a little-endian byte stream.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        use crate::gguf::value::read_string;

        let name = read_string(reader)?;
        let n_dims = reader.read_u32::<LittleEndian>()?;
        if n_dims > 4 {
            return Err(Error::InvalidShape { n_dims });
        }

        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(reader.read_u64::<LittleEndian>()?);
        }

        let ty = TensorType::from_u32(reader.read_u32::<LittleEndian>()?)?;
        let offset = reader.read_u64::<LittleEndian>()?;

        Ok(Self {
            name,
            shape,
            ty,
            offset,
        })
    }
}
