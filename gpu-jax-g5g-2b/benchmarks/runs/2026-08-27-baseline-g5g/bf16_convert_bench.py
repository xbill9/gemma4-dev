"""Which bf16 -> float16 host conversion is actually fastest on Graviton2?

docs/bf16-weights-on-turing.md records `ndarray.astype(float16)` on ml_dtypes
bfloat16 as unusably slow -- E2B's 4.7 GB table "did not finish in 10 minutes"
(2026-08-24) -- and names a view(uint16) bit-shift as the untried alternative.
On an x86 dev box in 2026-08-27 the direct astype ran at ~1.3 GB/s and the
bit-shift was 10x SLOWER, so the premise has to be re-measured on the chip it
was measured on rather than inherited.
"""
import numpy as np, ml_dtypes, time, platform
bf16 = ml_dtypes.bfloat16
print("machine:", platform.machine(), "numpy", np.__version__, "ml_dtypes", ml_dtypes.__version__)

def bitshift(arr, chunk=1 << 22):
    flat = arr.reshape(-1).view(np.uint16)
    out = np.empty(flat.size, dtype=np.float16)
    for i in range(0, flat.size, chunk):
        u32 = flat[i:i + chunk].astype(np.uint32) << np.uint32(16)
        out[i:i + chunk] = u32.view(np.float32).astype(np.float16)
    return out.reshape(arr.shape)

def via_f32(arr):  return arr.astype(np.float32).astype(np.float16)
def direct(arr):   return arr.astype(np.float16)

rng = np.random.default_rng(0)
src = rng.standard_normal((8192, 4096)).astype(np.float32).astype(bf16)   # 64 MiB
mb = src.nbytes / 1e6
ref = None
for name, fn in (("direct astype", direct), ("via float32", via_f32), ("bitshift", bitshift)):
    try:
        t = time.time(); got = fn(src); dt = time.time() - t
        if ref is None: ref = got
        same = np.array_equal(got.view(np.uint16), ref.view(np.uint16))
        print(f"  {name:16s} {dt:7.3f}s  {mb/dt/1000:6.2f} GB/s  bit-identical={same}"
              f"   -> 4.70 GB table would take {4700/ (mb/dt):6.1f}s")
    except Exception as e:
        print(f"  {name:16s} FAILED {type(e).__name__}: {e}")
