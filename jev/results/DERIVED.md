# Derived figures for the measurement article (computed by derived_figures.py)

- 26B minus E4B, four tasks, points: {'sst2': 0.7, 'ag_news': 3.0, 'emotion': 4.3, 'irony': 5.4}; range 0.7 to 5.4
- Jev at $0.042 per million input tokens: $5.54 per million decisions at the median 132 tokens, $12.56 at the longest 299
- Plain against DiffusionGemma, time per decision: one read 1.9 to 2.0 times plain, four reads 5.0 to 5.1 times plain
- Four reads against one read, this run: 2.6 to 2.6 times
- Mastracci, automatic re-reads against one read over his 8 sets: 1.9 to 4.6 times
- aegis2: Jev minus plain +2.0 points, 95% range -5.1 to +9.1
- boolq: Jev minus plain +5.7 points, 95% range +0.3 to +11.1
- civil_comments: Jev minus plain -6.0 points, 95% range -11.8 to -0.2
- paws: Jev minus plain +5.2 points, 95% range -0.8 to +11.2
- squad2: Jev minus plain -6.4 points, 95% range -11.9 to -0.9
- After 50 labels, DiffusionGemma's ECE below plain's on 8 of 13 subsets; two-sided sign test p = 0.58
- All three runs: $3.20
