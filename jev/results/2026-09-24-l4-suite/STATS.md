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

Published Nimble-9B accuracy, pooled: all 75.9%, noul 80.1%, choice 81.1%, score 51.2%

Jev minus arm, points, unpaired 95% range:
- all: gemma-4-26B-A4B-it-AWQ-4bit +2.1 (+0.2 to +4.0); diffusiongemma-26B-A4B-it-AWQ-INT4 +1.4 (-0.4 to +3.3); gemma-4-E4B-it +4.1 (+2.2 to +6.0)
- noul: gemma-4-26B-A4B-it-AWQ-4bit -0.1 (-2.8 to +2.5); diffusiongemma-26B-A4B-it-AWQ-INT4 +0.1 (-2.5 to +2.8); gemma-4-E4B-it +2.7 (-0.0 to +5.5)
- choice: gemma-4-26B-A4B-it-AWQ-4bit +4.5 (+2.0 to +7.1); diffusiongemma-26B-A4B-it-AWQ-INT4 +5.5 (+3.0 to +8.1); gemma-4-E4B-it +8.0 (+5.4 to +10.6)
- score: gemma-4-26B-A4B-it-AWQ-4bit -0.3 (-5.8 to +5.2); diffusiongemma-26B-A4B-it-AWQ-INT4 -7.6 (-13.1 to -2.1); gemma-4-E4B-it -4.4 (-9.9 to +1.1)

Brier, median over 13 subsets (published Jev 0.267, Nimble 0.314):
- gemma-4-26B-A4B-it-AWQ-4bit: 0.359, below Jev on 2 of 13
- diffusiongemma-26B-A4B-it-AWQ-INT4: 0.290, below Jev on 4 of 13
- gemma-4-E4B-it: 0.359, below Jev on 2 of 13

DiffusionGemma raw ECE below plain on 13 of 13; after 50 labels on 8 of 13
Plain after 50 labels above Jev as shipped on 8 of 13, by up to 0.049
noul subsets, Jev minus plain, points: aegis2 +2.0, boolq +5.7, civil_comments -6.0, paws +5.2, squad2 -6.4
choice subsets, Jev minus plain, points: massive-de-DE +2.9, massive-en-US +1.1, multinli +1.3, pubmedqa +13.2, vitaminc-dev +5.5
choice subsets, Jev minus plain, records: massive-de-DE 10, massive-en-US 4, multinli 4, pubmedqa 33, vitaminc-dev 33
26B minus E4B per subset, points: from -9.6 (summeval-relevance) to +8.0 (boolq)
