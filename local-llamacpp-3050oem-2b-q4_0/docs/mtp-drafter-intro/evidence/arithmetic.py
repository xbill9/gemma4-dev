"""ARITHMETIC for the drafter intro. Inputs from drafter-q4exact-REPORT.md."""
e2b = dict(q8=97836704, q4=59236000, bf16=170213024)
e4b = dict(q8=98672288, q4=59678368, bf16=171785888)
for n, d in (("E2B", e2b), ("E4B", e4b)):
    print(f"{n}: q4/q8 {d['q4']/d['q8']:.3f} ({1-d['q4']/d['q8']:.0%} smaller); q4/bf16 {d['q4']/d['bf16']:.3f} ({1-d['q4']/d['bf16']:.0%} smaller)")
    print(f"{n}: MB q8 {d['q8']/1e6:.1f} q4 {d['q4']/1e6:.1f} bf16 {d['bf16']/1e6:.1f}")
print(f"error ratio q8/q4: {0.00494/0.00138:.1f}x")
print(f"E2B target VRAM -c 8192 none 2795, +q4 drafter 2875 = +{2875-2795} MiB; +q8 2915 = +{2915-2795}; +bf16 2983 = +{2983-2795}")
print(f"E4B none 4415, +q4 4499 = +{4499-4415}; +q8 4537 = +{4537-4415}; +bf16 4607 = +{4607-4415}")
