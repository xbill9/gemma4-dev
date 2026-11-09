## E2B — `google/gemma-4-E2B-it`

| Field | Value |
| :--- | ---: |
| `num_hidden_layers` | 35 |
| `num_kv_shared_layers` | 20 |
| `num_attention_heads` | 8 |
| `num_key_value_heads` | **1** |
| `head_dim` | 256 |
| `global_head_dim` | 512 |
| `hidden_size` | 1536 |
| `intermediate_size` | 6144 |
| `vocab_size` | 262,144 |
| `sliding_window` | 512 |
| `layer_types` | 4x `sliding_attention`, 1x `full_attention`, repeating (period 5) |

Also `tie_word_embeddings=True`, `final_logit_softcapping=30.0`, and per-layer embeddings
