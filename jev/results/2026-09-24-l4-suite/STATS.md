# Public-suite figures for the article

Readouts: {'gemma-4-26B-A4B-it-AWQ-4bit': 'ar', 'diffusiongemma-26B-A4B-it-AWQ-INT4': 'dg4', 'gemma-4-E4B-it': 'ar'}

| Group | n | Jev (published) | gemma-4-26B-A4B-it-AWQ-4bit | diffusiongemma-26B-A4B-it-AWQ-INT4 | gemma-4-E4B-it |
|---|---|---|---|---|---|
| all | 3880 | 77.3% (76.0–78.6) | 75.3% (73.9–76.6) | 75.9% (74.5–77.2) | 73.3% (71.9–74.6) |
| noul | 1399 | 84.6% (82.6–86.4) | 84.8% (82.8–86.6) | 84.5% (82.5–86.3) | 81.9% (79.8–83.8) |
| choice | 1848 | 82.8% (81.1–84.5) | 78.3% (76.4–80.1) | 77.3% (75.4–79.2) | 74.8% (72.8–76.8) |
| score | 633 | 45.2% (41.3–49.1) | 45.5% (41.7–49.4) | 52.8% (48.9–56.6) | 49.6% (45.7–53.5) |

Plain vs DiffusionGemma, paired per record:

- all: plain only right 187, diffusion only right 211, exact McNemar p = 0.2489
- noul: plain only right 43, diffusion only right 39, exact McNemar p = 0.7407
- choice: plain only right 93, diffusion only right 75, exact McNemar p = 0.1895
- score: plain only right 51, diffusion only right 97, exact McNemar p = 0.0002

ECE, median over 13 subsets:

- jev: {"raw_median": 0.071}
- nimble: {"raw_median": 0.109}
- gemma-4-26B-A4B-it-AWQ-4bit: {"raw_median": 0.18, "fit50_median": 0.08, "fit50_at_or_below_jev": 5, "raw_at_or_below_jev": 0}
- diffusiongemma-26B-A4B-it-AWQ-INT4: {"raw_median": 0.114, "fit50_median": 0.074, "fit50_at_or_below_jev": 6, "raw_at_or_below_jev": 2}
- gemma-4-E4B-it: {"raw_median": 0.173, "fit50_median": 0.077, "fit50_at_or_below_jev": 4, "raw_at_or_below_jev": 1}

Median label mass per read: diffusiongemma-26B-A4B-it-AWQ-INT4 66.5%, gemma-4-26B-A4B-it-AWQ-4bit 100.0%

Largest multiple-choice gap, plain Gemma against Jev: pubmedqa, Jev 77.2% against 64.0%
