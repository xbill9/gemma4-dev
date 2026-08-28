//! Gemma 4 E2B configuration, ported from `ports/gemma4/jax_e_model.py`.
//!
//! **Every shape-critical field is required.** The Python `Gemma4EConfig` carries dataclass
//! defaults that are *not* E2B's — `hidden_size` defaults to 2048 against E2B's real 1536, and
//! `num_key_value_heads` to 4 against E2B's real 1 — so a field missing from `config.json`
//! there yields a wrong-shaped model rather than an error. That is the single most dangerous
//! property of the Python port, and it is the one thing this port deliberately does not
//! reproduce: serde has no `#[serde(default)]` on any shape field, so a missing key is a parse
//! error naming the field.
//!
//! Only genuinely optional fields carry defaults, and each is commented with why.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4EConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    /// Sliding-attention head dim (E2B: 256).
    pub head_dim: usize,
    /// Full-attention head dim (E2B: 512). Applies to Q, K, V *and* the norms — not Q-only.
    pub global_head_dim: usize,
    pub num_global_key_value_heads: usize,

    /// MLP width. E2B: 6144.
    pub intermediate_size: usize,

    pub sliding_window: Option<usize>,
    /// Number of trailing layers that read an earlier layer's KV (E2B: 20 of 35).
    pub num_kv_shared_layers: usize,

    pub rope_theta: f64,
    pub global_rope_theta: f64,
    pub vocab_size: usize,

    /// `false` on E2B and E4B, `true` on 12B/26B/31B: full-attention layers ship no `v_proj`
    /// and one projection feeds both K and V. Defaulting to `false` is safe because a
    /// checkpoint that needs it says so; a loader tolerating a missing tensor instead would
    /// produce a silently broken model that still emits fluent text.
    #[serde(default)]
    pub attention_k_eq_v: bool,

    /// Per-layer embeddings (PLE). `Option` rather than defaulted: a checkpoint
    /// without PLE is a real thing, and absence must stay visible instead of being
    /// silently filled in with E2B's numbers.
    #[serde(default)]
    pub hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    pub vocab_size_per_layer_input: Option<usize>,

    /// E2B ties input and output embeddings.
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    /// E2B: 30.0.
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,

    /// Absent in most checkpoints; `layer_types()` derives the E2B period-5 pattern.
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
}

fn default_true() -> bool {
    true
}

pub const SLIDING: &str = "sliding_attention";
pub const FULL: &str = "full_attention";

impl Gemma4EConfig {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    /// Per-layer attention type. Mirrors `__post_init__`: sliding unless `i % 5 == 4`.
    pub fn layer_types(&self) -> Vec<String> {
        if let Some(lt) = &self.layer_types {
            return lt.clone();
        }
        (0..self.num_hidden_layers)
            .map(|i| if i % 5 != 4 { SLIDING } else { FULL }.to_string())
            .collect()
    }

    /// Layers at or above this index share an earlier layer's KV. E2B: 35 - 20 = 15.
    pub fn first_kv_shared_layer_idx(&self) -> usize {
        self.num_hidden_layers - self.num_kv_shared_layers
    }

    /// Maps each layer to the layer whose KV it uses: "the last preceding layer of the same
    /// attention type". With the period-5 pattern this collapses to exactly two sources.
    pub fn kv_share_map(&self) -> Vec<usize> {
        let types = self.layer_types();
        let first = self.first_kv_shared_layer_idx();
        let mut last_sliding: Option<usize> = None;
        let mut last_full: Option<usize> = None;
        for (i, t) in types.iter().enumerate().take(first) {
            if t == SLIDING {
                last_sliding = Some(i);
            } else {
                last_full = Some(i);
            }
        }
        (0..self.num_hidden_layers)
            .map(|i| {
                if i < first {
                    i
                } else if types[i] == SLIDING {
                    last_sliding.expect("no preceding sliding layer to share from")
                } else {
                    last_full.expect("no preceding full-attention layer to share from")
                }
            })
            .collect()
    }

    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types()[layer] == SLIDING
    }

    /// head_dim for a layer. **Never derive this as `hidden_size / num_attention_heads`** —
    /// on E2B that gives 1536/8 = 192 and is wrong everywhere. The heads do not tile
    /// hidden_size: 8 x 256 = 2048 against a hidden_size of 1536.
    pub fn head_dim_for(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.head_dim
        } else {
            self.global_head_dim
        }
    }

    pub fn kv_heads_for(&self, layer: usize) -> usize {
        if self.is_sliding(layer) {
            self.num_key_value_heads
        } else {
            self.num_global_key_value_heads
        }
    }

    /// `q_proj` weight shape as `[out, in]`, the layout the safetensors headers use.
    /// E2B: 2048x1536 on sliding layers, 4096x1536 on full-attention ones.
    pub fn q_proj_shape(&self, layer: usize) -> [usize; 2] {
        [
            self.num_attention_heads * self.head_dim_for(layer),
            self.hidden_size,
        ]
    }

    /// `k_proj` / `v_proj` weight shape as `[out, in]`.
    ///
    /// E2B: **256**x1536 sliding and **512**x1536 full — one KV head, so `out` equals
    /// head_dim. This is the cheapest check that a config is really E2B: with
    /// `num_key_value_heads = 4` the sliding `k_proj` would be 1024x1536, and that is
    /// exactly the error `MODELS.md` records as having propagated from a note
    /// "filled in from memory".
    pub fn kv_proj_shape(&self, layer: usize) -> [usize; 2] {
        [
            self.kv_heads_for(layer) * self.head_dim_for(layer),
            self.hidden_size,
        ]
    }
}
