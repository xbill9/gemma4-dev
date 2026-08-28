//! Gemma 4 E2B model geometry in Rust, for the `gpu-jaxrust-g5g-2b` rig.
//!
//! **Scope: this is the first increment of the port described in
//! `docs/rust-jax-runtime-survey.md`, and it is deliberately the part that needs no GPU.**
//! Config parsing, KV geometry and dtype conversion are host-side and architecture-independent,
//! so they can be written and tested before the `xla` crate, the aarch64 CUDA extension, or a
//! T4G is involved at all. The survey's recommendation was to port bottom-up against a parity
//! harness rather than top-down; this is the bottom.
//!
//! **What is NOT here:** the forward pass, attention, RoPE, the LM head, the loader's
//! safetensors parsing, and every XLA op. Those are the remaining ~3,700 lines.
//!
//! Every test in this crate asserts a number that was independently derived or measured on the
//! Python rig, so a divergence is a real signal rather than a restatement.

pub mod config;
pub mod dtype;
pub mod kv;

#[cfg(test)]
mod tests {
    use crate::config::{Gemma4EConfig, FULL, SLIDING};
    use crate::dtype::*;
    use crate::kv::*;

    /// E2B's real geometry, transcribed from the canonical table in the monorepo
    /// root `MODELS.md`, which was **verified by reading the safetensors headers**
    /// rather than a config file or a note.
    ///
    /// **Two of these values are actively contested in this repo and the table
    /// wins.** `MODELS.md` records that a 12B exploration note listed E2B at
    /// `num_key_value_heads = 4` and `num_global_key_value_heads = 4`, filled in
    /// from memory; the safetensors headers say **1**, and boot-time allocation
    /// arithmetic agrees to 0.1% on two chip generations. The sibling rig's
    /// `ports/gemma4/jax_e_smoke_test.py` still carries the 4/4 values — together
    /// with `hidden_size = 2048` — under the comment "Real Gemma 4 E2B MatFormer
    /// configuration". Those are the Python dataclass **defaults**, not E2B.
    /// `projection_shapes_match_the_safetensors_headers` is what catches it.
    fn e2b() -> Gemma4EConfig {
        Gemma4EConfig::from_json(
            r#"{
                "hidden_size": 1536,
                "intermediate_size": 6144,
                "num_hidden_layers": 35,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_global_key_value_heads": 1,
                "sliding_window": 512,
                "num_kv_shared_layers": 20,
                "rope_theta": 10000.0,
                "global_rope_theta": 1000000.0,
                "vocab_size": 262144,
                "hidden_size_per_layer_input": 256,
                "vocab_size_per_layer_input": 262144,
                "tie_word_embeddings": true,
                "final_logit_softcapping": 30.0
            }"#,
        )
        .expect("E2B config must parse")
    }

    #[test]
    fn a_missing_shape_field_is_an_error_not_a_default() {
        // The Python port would silently use hidden_size=2048 here. This must refuse.
        let err = Gemma4EConfig::from_json(
            r#"{"num_hidden_layers":35,"num_attention_heads":8,"num_key_value_heads":1,
                "head_dim":256,"global_head_dim":512,"num_global_key_value_heads":1,
                "num_kv_shared_layers":20,"rope_theta":1.0,"global_rope_theta":1.0,
                "intermediate_size":6144,"vocab_size":262144}"#,
        );
        assert!(err.is_err(), "missing hidden_size must not default");
        assert!(err.unwrap_err().to_string().contains("hidden_size"));
    }

    #[test]
    fn head_dim_must_not_be_derived_from_hidden_size() {
        let c = e2b();
        // The trap: 1536 / 8 = 192, which is wrong everywhere.
        assert_eq!(c.hidden_size / c.num_attention_heads, 192);
        assert_eq!(c.head_dim_for(0), 256);
        // Heads do not tile hidden_size.
        assert_eq!(c.num_attention_heads * c.head_dim, 2048);
        assert_ne!(c.num_attention_heads * c.head_dim, c.hidden_size);
    }

    #[test]
    fn two_attention_geometries_period_five() {
        let c = e2b();
        let t = c.layer_types();
        assert_eq!(t.len(), 35);
        assert_eq!(t.iter().filter(|x| *x == SLIDING).count(), 28);
        assert_eq!(t.iter().filter(|x| *x == FULL).count(), 7);
        assert_eq!(c.head_dim_for(0), 256);
        assert_eq!(c.head_dim_for(4), 512); // i % 5 == 4 -> full attention
                                            // 8:1 MQA, full MQA not GQA, on both geometries.
        assert_eq!(c.num_attention_heads / c.kv_heads_for(0), 8);
        assert_eq!(c.num_attention_heads / c.kv_heads_for(4), 8);
    }

    #[test]
    fn kv_sharing_collapses_to_exactly_two_sources() {
        let c = e2b();
        assert_eq!(c.first_kv_shared_layer_idx(), 15);
        let m = c.kv_share_map();
        assert_eq!(m.len(), 35);
        for (i, src) in m.iter().enumerate().take(15) {
            assert_eq!(*src, i, "owning layers map to themselves");
        }
        // Not a rolling window and not paired layers: layer 13 for every shared sliding
        // layer, layer 14 for every shared full one.
        let shared: std::collections::BTreeSet<usize> = m[15..].iter().copied().collect();
        assert_eq!(shared, [13usize, 14].into_iter().collect());
        assert_eq!(m[15..].iter().filter(|s| **s == 13).count(), 16);
        assert_eq!(m[15..].iter().filter(|s| **s == 14).count(), 4);
    }

    #[test]
    fn fifteen_caches_exist_not_thirty_five() {
        let c = e2b();
        let plan = plan_caches(&c, 4096, true);
        assert_eq!(plan.len(), 15);
        assert_eq!(plan.iter().filter(|p| p.is_sliding).count(), 12);
        assert_eq!(plan.iter().filter(|p| !p.is_sliding).count(), 3);
    }

    #[test]
    fn kv_is_free_on_this_rig() {
        let c = e2b();
        const MIB: usize = 1024 * 1024;
        // Reproduces CLAUDE.md's table exactly.
        assert_eq!(total_kv_bytes(&c, 4096, true), 30 * MIB);
        assert_eq!(total_kv_bytes(&c, 4096, false), 72 * MIB);
        assert_eq!(total_kv_bytes(&c, 8192, true), 54 * MIB);
        // 18 KiB/token, reproducing MODELS.md.
        assert_eq!(kv_bytes_per_token(&c), 18_432);

        // The whole point: the prefill transient this rig OOMs on is ~5.2 GiB, 174x larger.
        // Never size this rig's context from KV arithmetic.
        let prefill_transient = 5.2f64 * 1024.0 * 1024.0 * 1024.0;
        let ratio = prefill_transient / total_kv_bytes(&c, 4096, true) as f64;
        assert!(ratio > 170.0 && ratio < 180.0, "ratio was {ratio}");
    }

    #[test]
    fn padding_never_occupies_a_real_positions_slot() {
        // A 100-token prompt padded to bucket 512, against a 512-slot ring.
        let (real_len, bucket, buf) = (100usize, 512usize, 512usize);
        for p in real_len..bucket {
            assert_eq!(
                ring_slot(p, real_len, buf),
                None,
                "pad {p} must not be stored"
            );
        }
        for p in 0..real_len {
            assert_eq!(ring_slot(p, real_len, buf), Some(p % buf));
        }
        // Decode writes at prompt_len + t, not bucket + t. These differ, and using the
        // bucket is what overwrote real positions.
        assert_eq!(decode_write_slot(real_len, 0, buf), 100);
        assert_ne!(decode_write_slot(real_len, 0, buf), (bucket) % buf);
    }

    #[test]
    fn the_bucket_ladder_bounds_padding_below_the_sliding_window() {
        let window = 512usize;
        for len in [
            1usize, 63, 64, 65, 127, 200, 511, 512, 1000, 1415, 3515, 4096,
        ] {
            let b = bucket_for(len);
            assert!(b >= len);
            assert!(
                b - len < window,
                "len {len} -> bucket {b} pads {} which reaches the window",
                b - len
            );
        }
        // Worst case is 127, not B/2.
        let worst = (1usize..8192).map(|n| bucket_for(n) - n).max().unwrap();
        assert_eq!(worst, 127);
    }

    /// The safetensors table in `MODELS.md`, asserted shape for shape. Shapes are
    /// `[out, in]`, read off `model.language_model.layers.*.self_attn.*`.
    #[test]
    fn projection_shapes_match_the_safetensors_headers() {
        let c = e2b();
        // sliding_attention x28: q 2048x1536, k/v 256x1536, head_dim 256
        assert_eq!(c.q_proj_shape(0), [2048, 1536]);
        assert_eq!(c.kv_proj_shape(0), [256, 1536]);
        // full_attention x7: q 4096x1536, k/v 512x1536, head_dim 512
        assert_eq!(c.q_proj_shape(4), [4096, 1536]);
        assert_eq!(c.kv_proj_shape(4), [512, 1536]);
        assert_eq!(c.layer_types().iter().filter(|t| *t == SLIDING).count(), 28);
        assert_eq!(c.layer_types().iter().filter(|t| *t == FULL).count(), 7);
    }

    /// The documented wrong values, pinned so they cannot come back.
    ///
    /// `MODELS.md`: a 12B exploration note listed E2B at `num_key_value_heads` 4 and
    /// `num_global_key_value_heads` 4, filled in from memory; the safetensors headers
    /// say 1. The sibling rig's `jax_e_smoke_test.py` still uses 4/4 plus
    /// `hidden_size` 2048 under the comment "Real Gemma 4 E2B MatFormer
    /// configuration" — which are the Python dataclass defaults, not the checkpoint.
    #[test]
    fn the_four_kv_head_config_is_not_e2b_and_the_shapes_prove_it() {
        let wrong = Gemma4EConfig::from_json(
            r#"{
                "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 35,
                "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256,
                "global_head_dim": 512, "num_global_key_value_heads": 4,
                "sliding_window": 512, "num_kv_shared_layers": 20,
                "rope_theta": 10000.0, "global_rope_theta": 1000000.0, "vocab_size": 262144
            }"#,
        )
        .unwrap();
        // It parses — it is a valid config, just not this checkpoint's. The shapes
        // are what disagree with the headers, and by a factor of four.
        assert_eq!(wrong.kv_proj_shape(0), [1024, 2048]);
        assert_ne!(wrong.kv_proj_shape(0), e2b().kv_proj_shape(0));
        // And it would misprice KV by 4x, which is why it matters here.
        assert_eq!(kv_bytes_per_token(&wrong), 4 * kv_bytes_per_token(&e2b()));
    }

    #[test]
    fn ple_absence_stays_visible_rather_than_defaulting() {
        let c = e2b();
        assert_eq!(c.hidden_size_per_layer_input, Some(256));
        assert_eq!(c.vocab_size_per_layer_input, Some(262144));
        assert_eq!(c.final_logit_softcapping, Some(30.0));
        assert!(c.tie_word_embeddings);
        // A config without PLE reports None, not E2B's numbers.
        let no_ple = Gemma4EConfig::from_json(
            r#"{
                "hidden_size": 1536, "intermediate_size": 6144, "num_hidden_layers": 35,
                "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256,
                "global_head_dim": 512, "num_global_key_value_heads": 1,
                "sliding_window": 512, "num_kv_shared_layers": 20,
                "rope_theta": 1.0, "global_rope_theta": 1.0, "vocab_size": 262144
            }"#,
        )
        .unwrap();
        assert_eq!(no_ple.hidden_size_per_layer_input, None);
    }

    #[test]
    fn bf16_to_f32_is_exactly_a_sixteen_bit_shift() {
        for x in [0.0f32, 1.0, -1.0, 2.5, -0.125, 65504.0, 1e-8, f32::INFINITY] {
            let bits = f32_to_bf16_bits(x);
            let back = bf16_bits_to_f32(bits);
            assert_eq!(back.to_bits(), (bits as u32) << 16);
            assert_eq!(back, half::bf16::from_bits(bits).to_f32());
        }
    }

    #[test]
    fn shard_conversion_matches_elementwise_and_stores_no_bf16() {
        let vals: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 7.0).collect();
        let mut bytes = Vec::new();
        for v in &vals {
            bytes.extend_from_slice(&f32_to_bf16_bits(*v).to_le_bytes());
        }
        let mut out = Vec::new();
        convert_bf16_shard(&bytes, &mut out).unwrap();
        assert_eq!(out.len(), vals.len());
        for (i, v) in vals.iter().enumerate() {
            let expect = half::f16::from_f32(half::bf16::from_f32(*v).to_f32());
            assert_eq!(out[i], expect, "element {i}");
        }
        assert!(convert_bf16_shard(&[0u8; 3], &mut out).is_err());
    }
}
