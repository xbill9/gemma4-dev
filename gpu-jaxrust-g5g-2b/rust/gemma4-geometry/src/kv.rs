//! KV cache geometry and the ring-store invariant.
//!
//! This module is pure integer arithmetic and deliberately has no XLA dependency: it is the
//! part of the port that can be verified without a GPU, and it is where the expensive findings
//! live. `docs/padding-window-eviction.md` cost a week; the invariant it produced is asserted
//! here rather than left implicit in a write path.

use crate::config::Gemma4EConfig;

/// Bytes per element of the KV cache. Turing computes in float16 (no bf16 datapath), so 2.
pub const KV_ELEM_BYTES: usize = 2;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerCache {
    pub layer: usize,
    pub is_sliding: bool,
    pub slots: usize,
    pub head_dim: usize,
    pub kv_heads: usize,
}

impl LayerCache {
    /// K and V, so x2.
    pub fn bytes(&self) -> usize {
        self.slots * self.head_dim * self.kv_heads * 2 * KV_ELEM_BYTES
    }
}

/// Allocate caches for the *owning* layers only — `0..first_kv_shared_layer_idx`.
/// E2B allocates **15**, not 35: the other 20 read an earlier layer's.
///
/// `window_kv` caps sliding layers at `sliding_window` slots. It auto-resolves to true whenever
/// `max_seq_len > sliding_window`, which is always the case on this rig.
pub fn plan_caches(cfg: &Gemma4EConfig, max_seq_len: usize, window_kv: bool) -> Vec<LayerCache> {
    (0..cfg.first_kv_shared_layer_idx())
        .map(|i| {
            let is_sliding = cfg.is_sliding(i);
            let slots = match (window_kv, is_sliding, cfg.sliding_window) {
                (true, true, Some(w)) => max_seq_len.min(w),
                _ => max_seq_len,
            };
            LayerCache {
                layer: i,
                is_sliding,
                slots,
                head_dim: cfg.head_dim_for(i),
                kv_heads: cfg.kv_heads_for(i),
            }
        })
        .collect()
}

pub fn total_kv_bytes(cfg: &Gemma4EConfig, max_seq_len: usize, window_kv: bool) -> usize {
    plan_caches(cfg, max_seq_len, window_kv)
        .iter()
        .map(|c| c.bytes())
        .sum()
}

/// Bytes of KV added per token *if every cached layer grew* — the figure `MODELS.md` quotes.
/// It is not what a sliding layer actually costs once the ring saturates.
pub fn kv_bytes_per_token(cfg: &Gemma4EConfig) -> usize {
    (0..cfg.first_kv_shared_layer_idx())
        .map(|i| cfg.head_dim_for(i) * cfg.kv_heads_for(i) * 2 * KV_ELEM_BYTES)
        .sum()
}

/// **The invariant, made executable.**
///
/// A cache index is an absolute *real* position, and padding never occupies an index a real
/// position uses. Right-padding to a bucket and writing pad K/V into the ring is what evicted
/// real tokens: at `pad_len >= sliding_window` the ring held only padding, 28 of E2B's 35
/// layers attended to an entirely masked window, and the model emitted a token loop the server
/// recorded as `status="success"`.
///
/// Returns the ring slot for an absolute position, or `None` if the position is padding and
/// must not be stored at all. Masking cannot repair this — the gap has to be removed, not
/// skipped.
pub fn ring_slot(position: usize, real_len: usize, buf_len: usize) -> Option<usize> {
    if position >= real_len {
        return None; // padding: never occupies a slot
    }
    Some(position % buf_len)
}

/// Decode writes at `prompt_len + t`, never at `bucket + t`. Using the padded bucket is what
/// put generated tokens at indices the real prompt already owned.
pub fn decode_write_slot(prompt_len: usize, t: usize, buf_len: usize) -> usize {
    (prompt_len + t) % buf_len
}

/// Worst-case padding for the bucket ladder: `(64, 128, 256)` then 128-steps to 16384.
/// Chosen so worst-case padding is **127** tokens rather than `B/2`, keeping `pad_len` below
/// every `sliding_window` Gemma 4 declares (E2B: 512) and the eviction failure unreachable.
pub fn bucket_for(seq_len: usize) -> usize {
    for b in [64usize, 128, 256] {
        if seq_len <= b {
            return b;
        }
    }
    let mut b = 384;
    while b < 16384 {
        if seq_len <= b {
            return b;
        }
        b += 128;
    }
    16384
}
