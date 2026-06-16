use crate::gguf::error::{Error, Result};
use crate::gguf::file::GgufFile;
use crate::gguf::tensor::TensorInfo;
use memmap2::Mmap;
use std::fs::File;
use std::io::Cursor;
use std::path::Path;

/// Memory-mapped view of a GGUF file's tensor weight data.
///
/// `Weights` owns both the parsed file metadata and the underlying `mmap` of the
/// file. Tensor bytes are returned as zero-copy slices into the mapping.
#[derive(Debug)]
pub struct Weights {
    file: GgufFile,
    mmap: Mmap,
}

impl Weights {
    /// Open a GGUF file and memory-map its contents.
    ///
    /// The entire file is mapped; raw tensor bytes are returned as sub-slices of
    /// the mapping, so no weight data is copied during lookup.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        // Safety: we map a file we just opened read-only. We do not mutate the
        // underlying file through this mapping, and aliasing is managed by the
        // OS read-only mapping semantics.
        let mmap = unsafe { Mmap::map(&file)? };
        let gguf = GgufFile::read(Cursor::new(&mmap[..]))?;
        Ok(Self { file: gguf, mmap })
    }

    /// Access the parsed GGUF metadata and tensor info table.
    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    /// Return the raw bytes for the tensor with the given name.
    ///
    /// Returns `Err` if the tensor does not exist, its type size is unknown, or
    /// its byte range falls outside the mapped file.
    pub fn tensor(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .file
            .tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        self.tensor_bytes(info)
    }

    /// Dequantize the tensor with the given name to a `Vec<f32>`.
    ///
    /// Supported types: F32, F16, Q4_0, Q8_0.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .file
            .tensor(name)
            .ok_or_else(|| Error::TensorNotFound(name.to_string()))?;
        let bytes = self.tensor_bytes(info)?;
        crate::gguf::dequant::dequantize_to_vec(info, bytes)
    }

    /// Dequantize the tensor at the given info-table index to a `Vec<f32>`.
    ///
    /// Supported types: F32, F16, Q4_0, Q8_0.
    pub fn tensor_f32_by_index(&self, index: usize) -> Result<Vec<f32>> {
        let info = self
            .file
            .tensors
            .get(index)
            .ok_or(Error::InvalidTensorIndex(index))?;
        let bytes = self.tensor_bytes(info)?;
        crate::gguf::dequant::dequantize_to_vec(info, bytes)
    }

    /// Return the raw bytes for the tensor at the given index in the info table.
    pub fn tensor_by_index(&self, index: usize) -> Result<&[u8]> {
        let info = self
            .file
            .tensors
            .get(index)
            .ok_or(Error::InvalidTensorIndex(index))?;
        self.tensor_bytes(info)
    }

    /// Iterate over `(name, raw_bytes)` for every tensor in the file.
    ///
    /// If a tensor cannot be sliced (unknown type or out-of-bounds), the
    /// iterator yields `Err` for that entry and continues.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, Result<&[u8]>)> {
        self.file.tensors.iter().map(|info| {
            let bytes = self.tensor_bytes(info);
            (info.name.as_str(), bytes)
        })
    }

    fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8]> {
        let size = info
            .byte_size()
            .ok_or(Error::UnknownTensorType(info.ty as u32))?;
        let start: usize = info
            .data_offset(self.file.tensor_data_offset)
            .try_into()
            .map_err(|_| Error::TensorOffsetOverflow)?;
        let end = start
            .checked_add(size)
            .ok_or(Error::TensorOffsetOverflow)?;
        if end > self.mmap.len() {
            return Err(Error::UnexpectedEof);
        }
        Ok(&self.mmap[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, WriteBytesExt};
    use std::io::Write;

    fn write_test_gguf(path: &Path) {
        let mut buf: Vec<u8> = Vec::new();

        // Header
        buf.write_all(b"GGUF").unwrap();
        buf.write_u32::<LittleEndian>(3).unwrap(); // version
        buf.write_u64::<LittleEndian>(1).unwrap(); // n_tensors
        buf.write_u64::<LittleEndian>(0).unwrap(); // n_kv

        // Tensor info table
        let name = b"token_embeddings";
        buf.write_u64::<LittleEndian>(name.len() as u64).unwrap();
        buf.write_all(name).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap(); // n_dims
        buf.write_u64::<LittleEndian>(2).unwrap(); // shape[0]
        buf.write_u32::<LittleEndian>(0).unwrap(); // type F32
        buf.write_u64::<LittleEndian>(0).unwrap(); // offset

        // Pad to 32-byte alignment
        while !buf.len().is_multiple_of(32) {
            buf.write_u8(0).unwrap();
        }

        // Tensor data: two f32 values
        buf.write_f32::<LittleEndian>(1.5).unwrap();
        buf.write_f32::<LittleEndian>(2.5).unwrap();

        std::fs::write(path, &buf).unwrap();
    }

    #[test]
    fn opens_and_reads_tensor_bytes() {
        let tmp = std::env::temp_dir().join("inferox_weights_test.gguf");
        write_test_gguf(&tmp);

        let weights = Weights::open(&tmp).unwrap();
        assert_eq!(weights.file().tensors.len(), 1);

        let bytes = weights.tensor("token_embeddings").unwrap();
        assert_eq!(bytes.len(), 8);

        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(floats, vec![1.5, 2.5]);

        std::fs::remove_file(&tmp).unwrap();
    }

    #[test]
    fn tensor_not_found_returns_error() {
        let tmp = std::env::temp_dir().join("inferox_weights_missing_test.gguf");
        write_test_gguf(&tmp);

        let weights = Weights::open(&tmp).unwrap();
        assert!(matches!(
            weights.tensor("missing"),
            Err(Error::TensorNotFound(_))
        ));

        std::fs::remove_file(&tmp).unwrap();
    }
}
