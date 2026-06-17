use crate::gguf::error::{Error, Result};
use crate::gguf::tensor::TensorInfo;
use crate::gguf::types::TensorType;
use half::f16;

/// Dequantize a tensor's raw bytes into a caller-provided `f32` buffer.
///
/// The output length must equal the tensor's element count.
/// Supported types: F32, F16, Q4_0, Q8_0, Q4_K.
pub fn dequantize_f32(info: &TensorInfo, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let expected = info
        .byte_size()
        .ok_or(Error::UnknownTensorType(info.ty as u32))?;
    if bytes.len() != expected {
        return Err(Error::UnexpectedEof);
    }
    if out.len() != info.n_elements() as usize {
        return Err(Error::InvalidTensorIndex(out.len()));
    }

    match info.ty {
        TensorType::F32 => dequantize_f32_f32(bytes, out),
        TensorType::F16 => dequantize_f16_f32(bytes, out),
        TensorType::Q4_0 => dequantize_q4_0(bytes, out),
        TensorType::Q8_0 => dequantize_q8_0(bytes, out),
        TensorType::Q4K => dequantize_q4_k(bytes, out),
        _ => Err(Error::UnknownTensorType(info.ty as u32)),
    }
}

/// Dequantize a tensor into a freshly allocated `Vec<f32>`.
pub fn dequantize_to_vec(info: &TensorInfo, bytes: &[u8]) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; info.n_elements() as usize];
    dequantize_f32(info, bytes, &mut out)?;
    Ok(out)
}

fn dequantize_f32_f32(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(())
}

fn dequantize_f16_f32(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        out[i] = f16::from_le_bytes(chunk.try_into().unwrap()).to_f32();
    }
    Ok(())
}

/// GGML Q4_0: 32 elements per block, 18 bytes per block.
///
/// Block layout:
///   - 2 bytes: scale `delta` as little-endian f16
///   - 16 bytes: 32 nibbles (low nibble = element j, high nibble = element j+16)
///
/// Each unsigned nibble `q` in 0..15 represents the signed value `q - 8`,
/// and the final value is `(q - 8) * delta`.
fn dequantize_q4_0(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 18;

    if !bytes.len().is_multiple_of(BLOCK_BYTES) {
        return Err(Error::UnexpectedEof);
    }

    for (block, chunk) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
        let delta = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        let qs = &chunk[2..];
        let base = block * BLOCK_ELEMS;

        for j in 0..BLOCK_ELEMS / 2 {
            let byte = qs[j];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = (byte >> 4) as i32 - 8;
            out[base + j] = q0 as f32 * delta;
            out[base + j + BLOCK_ELEMS / 2] = q1 as f32 * delta;
        }
    }
    Ok(())
}

/// GGML Q8_0: 32 elements per block, 34 bytes per block.
///
/// Block layout:
///   - 2 bytes: scale `delta` as little-endian f16
///   - 32 bytes: signed int8 quantized values
///
/// Dequantization: `y[j] = qs[j] * delta`.
fn dequantize_q8_0(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_ELEMS: usize = 32;
    const BLOCK_BYTES: usize = 34;

    if !bytes.len().is_multiple_of(BLOCK_BYTES) {
        return Err(Error::UnexpectedEof);
    }

    for (block, chunk) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
        let delta = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        let base = block * BLOCK_ELEMS;

        for j in 0..BLOCK_ELEMS {
            let q = chunk[2 + j] as i8 as f32;
            out[base + j] = q * delta;
        }
    }
    Ok(())
}

/// GGML Q4_K: 256 elements per block, 144 bytes per block.
///
/// Block layout:
///   - 2 bytes: global scale `d` as little-endian f16
///   - 2 bytes: global min `dmin` as little-endian f16
///   - 12 bytes: packed 6-bit scales and mins for 8 groups of 32 weights
///   - 128 bytes: 256 nibbles of quantized weights
///
/// Each group `j` has a 6-bit scale `sc` and 6-bit min `mn`. For weights in
/// that group, `y = d * sc * q - dmin * mn` where `q` is a nibble in 0..15.
fn dequantize_q4_k(bytes: &[u8], out: &mut [f32]) -> Result<()> {
    const BLOCK_ELEMS: usize = 256;
    const BLOCK_BYTES: usize = 144;
    const GROUP_ELEMS: usize = 32;
    const N_GROUPS: usize = BLOCK_ELEMS / GROUP_ELEMS;

    if !bytes.len().is_multiple_of(BLOCK_BYTES) {
        return Err(Error::UnexpectedEof);
    }

    for (block, chunk) in bytes.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        let dmin = f16::from_le_bytes([chunk[2], chunk[3]]).to_f32();
        let scales = &chunk[4..16];
        let qs = &chunk[16..];
        let base = block * BLOCK_ELEMS;

        for group in 0..N_GROUPS {
            let (sc, mn) = if group < 4 {
                let sc = (scales[group] & 0x3F) as f32;
                let mn = (scales[group + 4] & 0x3F) as f32;
                (sc, mn)
            } else {
                let sc = ((scales[group + 4] & 0x0F) as u32 | ((scales[group - 4] as u32 >> 6) << 4)) as f32;
                let mn = ((scales[group + 4] as u32 >> 4) | ((scales[group] as u32 >> 6) << 4)) as f32;
                (sc, mn)
            };

            let dall = d * sc;
            let dmin_val = dmin * mn;
            let qs_offset = group * (GROUP_ELEMS / 2);

            for l in 0..GROUP_ELEMS {
                let byte = qs[qs_offset + l / 2];
                let q = if l % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                out[base + group * GROUP_ELEMS + l] = dall * q as f32 - dmin_val;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_info(ty: TensorType, shape: &[u64], _n_elements: u64) -> TensorInfo {
        TensorInfo {
            name: "test".to_string(),
            shape: shape.to_vec(),
            ty,
            offset: 0,
        }
    }

    #[test]
    fn dequantize_q4_0_single_block() {
        let mut block = Vec::with_capacity(18);
        // delta = 0.5
        block.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
        // 16 bytes of nibbles: low nibble = j, high nibble = j
        // => each q = j, value = (j - 8) * 0.5
        for j in 0..16_u8 {
            let byte = (j << 4) | j;
            block.push(byte);
        }

        let info = tensor_info(TensorType::Q4_0, &[32], 32);
        let out = dequantize_to_vec(&info, &block).unwrap();

        assert_eq!(out.len(), 32);
        for (j, v) in out.iter().enumerate() {
            let expected = (((j % 16) as i32) - 8) as f32 * 0.5;
            assert!((v - expected).abs() < 1e-6, "j={j}: got {v}, expected {expected}");
        }
    }

    #[test]
    fn dequantize_q8_0_single_block() {
        let mut block = Vec::with_capacity(34);
        // delta = 0.25
        block.extend_from_slice(&f16::from_f32(0.25).to_le_bytes());
        // 32 signed int8 values: 0, 1, 2, ..., 31
        for j in 0..32_i8 {
            block.push(j as u8);
        }

        let info = tensor_info(TensorType::Q8_0, &[32], 32);
        let out = dequantize_to_vec(&info, &block).unwrap();

        assert_eq!(out.len(), 32);
        for (j, v) in out.iter().enumerate() {
            let expected = (j as f32) * 0.25;
            assert!((v - expected).abs() < 1e-6, "j={j}: got {v}, expected {expected}");
        }
    }

    #[test]
    fn dequantize_f16_to_f32() {
        let bytes: Vec<u8> = [f16::from_f32(1.0), f16::from_f32(2.0), f16::from_f32(-3.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let info = tensor_info(TensorType::F16, &[3], 3);
        let out = dequantize_to_vec(&info, &bytes).unwrap();
        assert_eq!(out, vec![1.0, 2.0, -3.0]);
    }

    #[test]
    fn dequantize_f32_identity() {
        let bytes: Vec<u8> = [1.0f32, 2.0f32, -3.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let info = tensor_info(TensorType::F32, &[3], 3);
        let out = dequantize_to_vec(&info, &bytes).unwrap();
        assert_eq!(out, vec![1.0, 2.0, -3.0]);
    }

    #[test]
    fn dequantize_q4_k_single_block() {
        let mut block = Vec::with_capacity(144);
        // d = 1.0, dmin = 0.0
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        // scales: groups 0-3 sc=1/mn=0, groups 4-7 sc=1/mn=0
        block.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        // qs: 128 bytes of nibbles 0,1,2,...,31 mod 16 repeating per group
        for _ in 0..8 {
            for pair in 0..16_u8 {
                let low = 2 * pair;
                let high = (2 * pair + 1) % 32;
                let byte = (high << 4) | low;
                block.push(byte);
            }
        }

        let info = tensor_info(TensorType::Q4K, &[256], 256);
        let out = dequantize_to_vec(&info, &block).unwrap();

        assert_eq!(out.len(), 256);
        for (j, v) in out.iter().enumerate() {
            let expected = ((j % 32) % 16) as f32;
            assert!((v - expected).abs() < 1e-6, "j={j}: got {v}, expected {expected}");
        }
    }

    #[test]
    fn unsupported_type_fails() {
        let info = tensor_info(TensorType::Q5_0, &[32], 32);
        let bytes = vec![0u8; 22]; // Q5_0 block size
        assert!(matches!(
            dequantize_to_vec(&info, &bytes),
            Err(Error::UnknownTensorType(_))
        ));
    }
}
