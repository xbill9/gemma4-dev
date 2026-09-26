# 26B QAT W4A16 article: derived figures

## Repack verify (verify_report.json)

- experts.down_proj: 237,895,680 groups, 0 off the grid, 92.6% bit-identical, worst 7.8e-03
- experts.gate_up_proj: 475,791,360 groups, 0 off the grid, 92.6% bit-identical, worst 7.8e-03
- mlp.down_proj: 5,575,680 groups, 0 off the grid, 92.0% bit-identical, worst 7.8e-03
- mlp.gate_proj: 5,575,680 groups, 0 off the grid, 92.3% bit-identical, worst 1.0e-02
- mlp.up_proj: 5,575,680 groups, 0 off the grid, 92.5% bit-identical, worst 7.8e-03
- self_attn.k_proj: 4,956,160 groups, 0 off the grid, 90.0% bit-identical, worst 1.1e-02
- self_attn.o_proj: 12,615,680 groups, 0 off the grid, 89.7% bit-identical, worst 1.1e-02
- self_attn.q_proj: 12,615,680 groups, 0 off the grid, 90.0% bit-identical, worst 1.1e-02
- self_attn.v_proj: 4,505,600 groups, 0 off the grid, 90.4% bit-identical, worst 1.1e-02
- all quantized tensors: 765,107,200 groups
- copied: 748 of 748 byte-identical

## Accuracy, W4A16 against FP8 (2026-09-26-moe2-VS-FP8.json)

### FP8 (ref) against TPU W4A16

- sst2: ref 94.7%, test 95.0%, difference +0.3 points (-0.7 to +1.7); 1 right->wrong, 2 wrong->right
- ag_news: ref 86.0%, test 86.0%, difference +0.0 points (-1.7 to +1.7); 4 right->wrong, 4 wrong->right
- emotion: ref 58.3%, test 60.0%, difference +1.7 points (-0.3 to +4.0); 3 right->wrong, 8 wrong->right
- irony: ref 91.0%, test 90.7%, difference -0.3 points (-2.3 to +2.0); 6 right->wrong, 5 wrong->right
- reversed options (3 choice tasks): ref 79.4%, test 79.8%, difference +0.3 points (-0.6 to +1.3); 8 right->wrong, 11 wrong->right
- suite (all subsets): ref 76.0%, test 75.3%, difference -0.7 points (-1.5 to +0.1); 133 right->wrong, 106 wrong->right

### TPU W4A16 (ref) against L4 W4A16

- sst2: ref 95.0%, test 94.7%, difference -0.3 points (-1.0 to +0.0); 1 right->wrong, 0 wrong->right
- ag_news: ref 86.0%, test 86.0%, difference +0.0 points (-1.0 to +1.0); 1 right->wrong, 1 wrong->right
- emotion: ref 60.0%, test 60.0%, difference +0.0 points (+0.0 to +0.0); 0 right->wrong, 0 wrong->right
- irony: ref 90.7%, test 91.0%, difference +0.3 points (+0.0 to +1.0); 0 right->wrong, 1 wrong->right
- reversed options (3 choice tasks): ref 79.8%, test 79.7%, difference -0.1 points (-0.6 to +0.2); 2 right->wrong, 1 wrong->right
- suite (all subsets): ref 75.3%, test 76.0%, difference +0.7 points (+0.3 to +1.1); 16 right->wrong, 42 wrong->right

### TPU default KV (ref) against explicit bf16 KV

- sst2: ref 95.0%, test 95.0%, difference +0.0 points (+0.0 to +0.0); 0 right->wrong, 0 wrong->right
- ag_news: ref 86.0%, test 86.0%, difference +0.0 points (+0.0 to +0.0); 0 right->wrong, 0 wrong->right
- emotion: ref 60.0%, test 60.0%, difference +0.0 points (+0.0 to +0.0); 0 right->wrong, 0 wrong->right
- irony: ref 90.7%, test 90.7%, difference +0.0 points (+0.0 to +0.0); 0 right->wrong, 0 wrong->right
- reversed options (3 choice tasks): ref 79.8%, test 79.8%, difference +0.0 points (+0.0 to +0.0); 0 right->wrong, 0 wrong->right
- suite (all subsets): ref 75.3%, test 75.5%, difference +0.2 points (-0.1 to +0.5); 13 right->wrong, 21 wrong->right


## Capacity and speed

- KV tokens: W4A16 53,888, FP8 3,456, ratio 15.6x
- output tok/s: W4A16 1283.3, FP8 668.4, ratio 1.92x

## Cost per million output tokens (arithmetic)

- flex-start $1.35/chip-hour, W4A16: $0.29
- flex-start $1.35/chip-hour, FP8: $0.56
- on-demand $2.97/chip-hour, W4A16: $0.64
- on-demand $2.97/chip-hour, FP8: $1.23
