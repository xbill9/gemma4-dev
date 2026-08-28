//! Weight dtype conversion — the largest single lever on this hardware.
//!
//! **86.8% of decode on the Python rig goes to a dtype mismatch**, measured
//! (`2026-08-27-baseline-xprof-g5g`): the loader stores every float parameter as bfloat16 while
//! the compute dtype is float16, so XLA inserts a convert in front of every use (54.0%) and the
//! matmuls that remain run as fp32 GEMV (32.8%). Turing has no bf16 datapath at all — bf16 does
//! not fail there, it *emulates* through fp32, which is why this hid for weeks.
//!
//! **So this port converts at load and never stores bf16.** That is the whole reason the
//! conversion lives at the bottom of the crate rather than in a helper.
//!
//! On the Python side the fix was blocked by a stale `ml_dtypes` whose element-wise cast was
//! unvectorised (E2B's 4.70 GB PLE table "did not finish in 10 minutes" on Graviton2). In Rust
//! the fast path is the *only* path and it needs no library: **bf16 -> f32 is exactly a 16-bit
//! left shift of the bit pattern**, so the conversion is a shift and a native f32 -> f16 cast.

use half::f16;

/// bfloat16 bits -> f32. Exact and total: bf16 is the top 16 bits of the f32 encoding, so this
/// is lossless for every input including NaN, infinities and subnormals.
#[inline]
pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 -> bfloat16 bits, round-to-nearest-even. Only needed to build test fixtures.
#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) as u16) | 0x0040; // quiet NaN
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    ((b.wrapping_add(rounding)) >> 16) as u16
}

/// bfloat16 bits -> f16, the dtype Turing actually computes in.
#[inline]
pub fn bf16_bits_to_f16(bits: u16) -> f16 {
    f16::from_f32(bf16_bits_to_f32(bits))
}

/// Convert a shard of bfloat16 weights in place into a float16 buffer.
///
/// Takes the source as raw little-endian bytes because that is how it arrives from safetensors,
/// and writes into a caller-owned destination so no bf16 copy outlives its converted form. The
/// Python attempts failed by holding source and destination resident together on a device whose
/// free memory is 66% fragmented; converting host-side into a caller's buffer is the shape that
/// avoids it.
pub fn convert_bf16_shard(src_le_bytes: &[u8], dst: &mut Vec<f16>) -> Result<(), String> {
    if !src_le_bytes.len().is_multiple_of(2) {
        return Err(format!(
            "bf16 shard has odd byte length {}",
            src_le_bytes.len()
        ));
    }
    dst.clear();
    dst.reserve(src_le_bytes.len() / 2);
    for c in src_le_bytes.chunks_exact(2) {
        dst.push(bf16_bits_to_f16(u16::from_le_bytes([c[0], c[1]])));
    }
    Ok(())
}
