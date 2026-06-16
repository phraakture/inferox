//! CPU compute kernels for inference.
//!
//! These are deliberately simple, correct reference implementations. They
//! operate on contiguous `f32` buffers and can be replaced with SIMD/BLAS
//! kernels later without changing call sites.

use std::f32;

/// Matrix multiply: `C = A @ B`.
///
/// All matrices are stored in row-major order:
/// - `A` has shape `[m, k]`
/// - `B` has shape `[k, n]`
/// - `C` has shape `[m, n]`
///
/// This is a naive triple-loop implementation. It is correct but not tuned.
pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "A shape mismatch");
    assert_eq!(b.len(), k * n, "B shape mismatch");
    assert_eq!(c.len(), m * n, "C shape mismatch");

    c.fill(0.0);

    for i in 0..m {
        for l in 0..k {
            let a_val = a[i * k + l];
            let b_row = &b[l * n..(l + 1) * n];
            let c_row = &mut c[i * n..(i + 1) * n];
            for j in 0..n {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

/// Matrix multiply with a transposed `B`: `C = A @ B.T`.
///
/// - `A` has shape `[m, k]`
/// - `B` has shape `[n, k]` (stored row-major, interpreted as transposed)
/// - `C` has shape `[m, n]`
///
/// This is common in transformer linear layers where weights are often stored
/// with the output dimension first.
pub fn matmul_t(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    assert_eq!(a.len(), m * k, "A shape mismatch");
    assert_eq!(b.len(), n * k, "B shape mismatch");
    assert_eq!(c.len(), m * n, "C shape mismatch");

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for l in 0..k {
                sum += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] = sum;
        }
    }
}

/// In-place RMSNorm: `x = x / sqrt(mean(x^2) + eps) * weight`.
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    assert_eq!(x.len(), weight.len(), "RMSNorm weight length mismatch");

    if x.is_empty() {
        return;
    }

    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();

    for (x_i, w_i) in x.iter_mut().zip(weight) {
        *x_i = *x_i * scale * w_i;
    }
}

/// In-place SiLU / Swish activation: `x = x * sigmoid(x)`.
pub fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu_scalar(*v);
    }
}

/// Elementwise SwiGLU: `out = silu(gate) * up`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len(), "SwiGLU gate/up length mismatch");
    assert_eq!(gate.len(), out.len(), "SwiGLU output length mismatch");

    for ((g, u), o) in gate.iter().zip(up).zip(out.iter_mut()) {
        *o = silu_scalar(*g) * u;
    }
}

#[inline]
fn silu_scalar(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    // Clamp to avoid overflow in exp() on large negative inputs.
    let x = x.clamp(-80.0, 80.0);
    1.0 / (1.0 + (-x).exp())
}

/// Elementwise addition with broadcasting: `out += bias`.
pub fn add_in_place(out: &mut [f32], bias: &[f32]) {
    assert_eq!(out.len(), bias.len(), "add length mismatch");
    for (o, b) in out.iter_mut().zip(bias) {
        *o += b;
    }
}

/// Softmax over the last dimension for a batch of rows.
///
/// `x` is `[rows, cols]` in row-major order. Softmax is applied independently
/// to each row in-place.
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize) {
    assert_eq!(x.len(), rows * cols, "softmax shape mismatch");

    for i in 0..rows {
        let row = &mut x[i * cols..(i + 1) * cols];

        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter_mut().map(|v| {
            *v = (*v - max).exp();
            *v
        }).sum();

        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Gather token embeddings by integer token ids.
///
/// `embeddings` is `[vocab_size, hidden_size]` in row-major order. For each
/// token id in `input_ids`, the corresponding row is copied into `out`, which
/// has shape `[input_ids.len(), hidden_size]`.
pub fn embedding_lookup(embeddings: &[f32], input_ids: &[u32], hidden_size: usize, out: &mut [f32]) {
    assert_eq!(out.len(), input_ids.len() * hidden_size, "embedding output shape mismatch");

    for (pos, &id) in input_ids.iter().enumerate() {
        let src = id as usize * hidden_size;
        let dst = pos * hidden_size;
        assert!(
            src + hidden_size <= embeddings.len(),
            "token id {id} out of vocabulary bounds"
        );
        out[dst..dst + hidden_size].copy_from_slice(&embeddings[src..src + hidden_size]);
    }
}

/// Apply Rotary Position Embeddings (RoPE) in-place.
///
/// `x` is `[seq_len, n_heads, head_dim]` in row-major order. The rotation is
/// applied to the first `rope_dim` dimensions of each head, using positions
/// `0..seq_len`.
pub fn rope(x: &mut [f32], seq_len: usize, n_heads: usize, head_dim: usize, rope_dim: usize, base: f32) {
    assert_eq!(x.len(), seq_len * n_heads * head_dim, "rope shape mismatch");
    assert!(rope_dim <= head_dim, "rope_dim must not exceed head_dim");
    assert!(rope_dim.is_multiple_of(2), "rope_dim must be even");

    for pos in 0..seq_len {
        for h in 0..n_heads {
            let head_offset = (pos * n_heads + h) * head_dim;
            for i in (0..rope_dim).step_by(2) {
                let idx = head_offset + i;
                let a = x[idx];
                let b = x[idx + 1];

                let inv_freq = 1.0 / base.powf(i as f32 / rope_dim as f32);
                let angle = pos as f32 * inv_freq;
                let (sin, cos) = angle.sin_cos();

                x[idx] = a * cos - b * sin;
                x[idx + 1] = a * sin + b * cos;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2x3x4() {
        // A = 2x3
        let a = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        // B = 3x4
        let b = vec![
            1.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 0.0, 0.0,
        ];
        let mut c = vec![0.0; 8];

        matmul(&a, &b, &mut c, 2, 4, 3);

        // Row 0: [1,2,3] @ B = [4,5,1,2]
        assert_eq!(&c[0..4], &[4.0, 5.0, 1.0, 2.0]);
        // Row 1: [4,5,6] @ B = [10,11,4,5]
        assert_eq!(&c[4..8], &[10.0, 11.0, 4.0, 5.0]);
    }

    #[test]
    fn matmul_t_matches_transpose() {
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![0.0, 1.0, 2.0, 3.0]; // 2x2, stored as if transposed -> [[0,1],[2,3]]
        let mut c = vec![0.0; 4];

        // B as stored is [[0,1],[2,3]]. B.T = [[0,2],[1,3]].
        // A @ B.T = [[1,2],[3,4]] @ [[0,2],[1,3]] = [[2,8],[4,18]]
        matmul_t(&a, &b, &mut c, 2, 2, 2);

        assert_eq!(c, vec![2.0, 8.0, 4.0, 18.0]);
    }

    #[test]
    fn rms_norm_unit_scale() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0; 4];
        rms_norm(&mut x, &weight, 1e-5);

        let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let scale = 1.0f32 / (mean_sq + 1e-5f32).sqrt();
        assert_eq!(x, vec![1.0 * scale, 2.0 * scale, 3.0 * scale, 4.0 * scale]);
    }

    #[test]
    fn rms_norm_with_weight() {
        let mut x = vec![2.0, 2.0, 2.0, 2.0];
        let weight = vec![0.5, 1.0, 1.5, 2.0];
        rms_norm(&mut x, &weight, 1e-5);

        // mean_sq = 4, scale ≈ 1/2, x_i ≈ 2 * 0.5 * weight_i = weight_i
        for (got, expected) in x.iter().zip(&weight) {
            assert!((got - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn silu_values() {
        let mut x = vec![0.0, 1.0, -1.0, 2.0];
        silu(&mut x);

        assert!((x[0] - 0.0).abs() < 1e-6);
        assert!((x[1] - (1.0 * sigmoid(1.0))).abs() < 1e-6);
        assert!((x[2] - (-sigmoid(-1.0))).abs() < 1e-6);
        assert!((x[3] - (2.0 * sigmoid(2.0))).abs() < 1e-6);
    }

    #[test]
    fn swiglu_elementwise() {
        let gate = vec![0.0, 1.0, -1.0];
        let up = vec![2.0, 3.0, 4.0];
        let mut out = vec![0.0; 3];

        swiglu(&gate, &up, &mut out);

        for i in 0..3 {
            let expected = silu_scalar(gate[i]) * up[i];
            assert!((out[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn add_in_place_works() {
        let mut out = vec![1.0, 2.0, 3.0];
        add_in_place(&mut out, &[0.5, 1.0, 1.5]);
        assert_eq!(out, vec![1.5, 3.0, 4.5]);
    }

    #[test]
    fn softmax_rows_sums_to_one() {
        let mut x = vec![
            1.0, 2.0, 3.0,
            0.0, 0.0, 0.0,
        ];
        softmax_rows(&mut x, 2, 3);

        let row0_sum: f32 = x[0..3].iter().sum();
        let row1_sum: f32 = x[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5);
        assert!((row1_sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }
}
