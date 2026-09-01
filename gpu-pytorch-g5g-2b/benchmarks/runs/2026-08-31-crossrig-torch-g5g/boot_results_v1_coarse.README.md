# `boot_results_v1_coarse.json` — the discarded first campaign

Kept as the citation for the resolution finding, not as a result.

This is the first boot campaign, run with poll intervals of 10 s (SSM) / 20 s (install) /
5 s (health). At that granularity two genuinely similar boots land on the same tick, and the
two PyTorch reps here report **health 214.4 s and install 125.1 s — identical to the tenth**,
which reads as perfect reproducibility and is really the measurement floor.

The data is not wrong: the campaign log independently shows 215 s and 216 s of wall clock. It
is unusably coarse. The campaign was discarded and re-run at 0.5 s health polling, and the very
next pair of boots came in at 216.90 s and 193.86 s — an 11.9% spread this file cannot see.

Superseded by `boot_results.json`. Do not quote these figures as results.
