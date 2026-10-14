# Answer quality with each drafter, RTX 3050 OEM

| model | task | A | B | n | A | B | B - A | 95% range | A only | B only | same text |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| e2b | gsm8k | none-1013 | none | 1319 | 0.898 | 0.898 | **+0.001** | -0.011 to +0.012 | 29 | 30 | 175 |
| e2b | gsm8k | none | q8 | 1319 | 0.898 | 0.899 | **+0.001** | -0.010 to +0.011 | 24 | 25 | 224 |
| e2b | gsm8k | none | q4x | 1319 | 0.898 | 0.902 | **+0.004** | -0.006 to +0.014 | 20 | 25 | 240 |
| e2b | gsm8k | q8 | q4x | 1319 | 0.899 | 0.902 | **+0.003** | -0.006 to +0.013 | 19 | 23 | 496 |
| e2b | bfcl_simple | none-1013 | none | 400 | 0.925 | 0.925 | **+0.000** | -0.010 to +0.010 | 2 | 2 | 389 |
| e2b | bfcl_simple | none | q8 | 400 | 0.925 | 0.922 | **-0.003** | -0.013 to +0.005 | 2 | 1 | 392 |
| e2b | bfcl_simple | none | q4x | 400 | 0.925 | 0.922 | **-0.003** | -0.013 to +0.005 | 2 | 1 | 391 |
| e2b | bfcl_simple | q8 | q4x | 400 | 0.922 | 0.922 | **+0.000** | +0.000 to +0.000 | 0 | 0 | 398 |
| e4b | gsm8k | none-1013 | none | 1319 | 0.933 | 0.934 | **+0.002** | -0.005 to +0.008 | 9 | 11 | 366 |
| e4b | gsm8k | none | q8 | 1319 | 0.934 | 0.931 | **-0.003** | -0.010 to +0.004 | 12 | 8 | 448 |
| e4b | gsm8k | none | q4x | 1319 | 0.934 | 0.933 | **-0.002** | -0.008 to +0.005 | 11 | 9 | 461 |
| e4b | gsm8k | q8 | q4x | 1319 | 0.931 | 0.933 | **+0.002** | -0.005 to +0.008 | 7 | 9 | 745 |
| e4b | bfcl_simple | none-1013 | none | 400 | 0.910 | 0.910 | **+0.000** | -0.010 to +0.010 | 2 | 2 | 386 |
| e4b | bfcl_simple | none | q8 | 400 | 0.910 | 0.915 | **+0.005** | +0.000 to +0.013 | 0 | 2 | 390 |
| e4b | bfcl_simple | none | q4x | 400 | 0.910 | 0.915 | **+0.005** | -0.005 to +0.015 | 1 | 3 | 391 |
| e4b | bfcl_simple | q8 | q4x | 400 | 0.915 | 0.915 | **+0.000** | -0.007 to +0.007 | 1 | 1 | 394 |

Wall time per task at 8 requests at once (gen_eval.py's wall_s, seconds):

| model | task | none | q8 | q4x | q8 / none | q4x / none |
|---|---|---:|---:|---:|---:|---:|
| e2b | gsm8k | 1766 | 1150 | 1167 | 1.54x faster | 1.51x faster |
| e2b | bfcl_simple | 79 | 66 | 67 | 1.19x faster | 1.18x faster |
| e4b | gsm8k | 2273 | 1342 | 1345 | 1.69x faster | 1.69x faster |
| e4b | bfcl_simple | 128 | 105 | 105 | 1.22x faster | 1.22x faster |

Drafter acceptance over both tasks (from /metrics):

| arm | drafted | accepted | acceptance | tokens generated |
|---|---:|---:|---:|---:|
| e2b-none | 0 | 0 | 0.0000 | 546155 |
| e2b-q8 | 414547 | 330559 | 0.7974 | 538910 |
| e2b-q4x | 416410 | 331950 | 0.7972 | 541236 |
| e4b-none | 0 | 0 | 0.0000 | 463839 |
| e4b-q8 | 346488 | 290162 | 0.8374 | 464586 |
| e4b-q4x | 344917 | 289094 | 0.8382 | 462734 |
