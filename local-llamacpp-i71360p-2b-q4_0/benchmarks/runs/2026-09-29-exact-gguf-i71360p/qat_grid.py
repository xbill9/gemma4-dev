"""For every >=2-D bf16 tensor: split the last dim into blocks of 32 and test
whether each block sits on a q4_0 grid (d = absmax/8, x/d integer in [-8,8])."""
import json, struct, collections, sys
import numpy as np
P = sys.argv[1]
f = open(P, 'rb'); n = struct.unpack('<Q', f.read(8))[0]; H = json.loads(f.read(n)); base = 8 + n
H.pop('__metadata__', None)
mm = np.memmap(P, dtype=np.uint8, mode='r')
rng = np.random.default_rng(0)
agg = collections.defaultdict(lambda: [0, 0, 0, 0, []])  # tensors, blocks, ongrid, le16, distinct
for k, v in H.items():
    if v['dtype'] != 'BF16' or len(v['shape']) < 2: continue
    s, e = v['data_offsets']
    u = np.frombuffer(mm[base + s: base + e], dtype=np.uint16)
    last = v['shape'][-1]
    if last % 32: 
        key = 'SKIP(last%32) ' + '.'.join('N' if p.isdigit() else p for p in k.split('.'))
        agg[key][0] += 1; continue
    blocks = u.reshape(-1, 32)
    idx = rng.choice(len(blocks), size=min(4000, len(blocks)), replace=False)
    b = (blocks[idx].astype(np.uint32) << 16).view(np.float32)
    amax = np.abs(b).max(1, keepdims=True); amax[amax == 0] = 1
    q = b / (amax / 8)
    on = (np.abs(q - np.rint(q)) < 0.03).all(1)
    dist = np.array([len(np.unique(r)) for r in b])
    key = '.'.join('N' if p.isdigit() else p for p in k.split('.'))
    a = agg[key]; a[0] += 1; a[1] += len(b); a[2] += on.sum(); a[3] += (dist <= 16).sum(); a[4].append(np.median(dist))
print(f"{'tensor group':78s} {'n':>3s} {'on q4_0 grid':>12s} {'<=16 distinct':>13s} {'median distinct':>15s}")
for key, (t, nb, on, le, md) in sorted(agg.items()):
    if nb == 0: print(f"{key:78s} {t:3d}"); continue
    print(f"{key:78s} {t:3d} {on/nb:12.1%} {le/nb:13.1%} {np.median(md):15.0f}")
