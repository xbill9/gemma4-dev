"""Per block of 32: step = smallest gap between distinct values; test all values are
integer multiples of step; report the integer range (bit width) and max distinct levels."""
import json, struct, collections, sys, re
import numpy as np
P = sys.argv[1]
f = open(P, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; H = json.loads(f.read(n)); base = 8 + n
H.pop('__metadata__', None)
mm = np.memmap(P, dtype=np.uint8, mode='r'); rng = np.random.default_rng(1)
want = re.compile(r'embed_tokens|layers\.0\.(mlp\.down_proj|self_attn\.q_proj)|audio_tower\.layers\.[05]\.(self_attn\.(q_proj|relative_k_proj)|feed_forward1\.ffw_layer_1|lconv1d\.linear_start)|vision_tower\.encoder\.layers\.0\.mlp\.up_proj')
print(f"{'tensor':72s} {'on lattice':>10s} {'int range':>10s} {'max levels':>10s} {'zero in lattice':>15s}")
for k, v in sorted(H.items()):
    if v['dtype'] != 'BF16' or len(v['shape']) < 2 or not want.search(k): continue
    s, e = v['data_offsets']
    blocks = np.frombuffer(mm[base+s:base+e], dtype=np.uint16).reshape(-1, 32)
    b = (blocks[rng.choice(len(blocks), 3000, replace=False)].astype(np.uint32) << 16).view(np.float32)
    ok = lo = hi = mx = 0; zero = 0; lo, hi = 99, -99
    for r in b:
        u = np.unique(r); mx = max(mx, len(u))
        if len(u) < 2: ok += 1; continue
        step = np.diff(u).min(); q = u / step
        if (np.abs(q - np.rint(q)) < 0.02 * np.maximum(1, np.abs(q))).all():
            ok += 1; lo = min(lo, int(np.rint(q).min())); hi = max(hi, int(np.rint(q).max()))
            zero += (0.0 in u)
    print(f"{k:72s} {ok/len(b):10.1%} {f'{lo}..{hi}':>10s} {mx:10d} {zero/len(b):15.1%}")
