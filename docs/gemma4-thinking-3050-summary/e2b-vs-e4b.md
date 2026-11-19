# Summary paper: E2B vs E4B (vs 12B) thinking on the RTX 3050 — saved material

Held back from the E4B article on 2026-11-18 at the user's request ("save the 2b vs 4b for a summary paper").
Every figure below was computed from the two runs' raw replies by `../gemma4-e4b-thinking-3050/evidence/derived.py`
(output `derived.txt` there) and audited by `evidence/audit_e2b_vs_e4b.py` here (32-claim E4B audit plus these).
Runs: `local-llamacpp-3050oem-2b-q4_0/benchmarks/runs/2026-11-16-thinking-e2b-3050oem` and
`local-llamacpp-3050oem-4b-q4_0/benchmarks/runs/2026-11-17-thinking-e4b-3050oem`. Add the 12B when wave 3 ends.

## Passages cut from the E4B draft, verbatim

**How E4B scores.** GSM8K 0.9272 to 0.9439 across the eight setups, and 9 to 13 of 30 on AIME 2025. Against E2B on the same problems with the same settings, E4B got 35 to 58 more GSM8K problems right, beyond chance in all eight setups (p = 0.00082 or smaller).
- 🟢 On a card that holds it, pick E4B over E2B for math: 35 to 58 more GSM8K problems right at 1.29 to 1.75 times the time.
#### Step 4 — E4B Against E2B

The E2B run used the same problems, scoring script, card, llama.cpp commit and settings, so each E4B cell pairs with its E2B twin problem by problem.

| Weights | Drafter | Thinking | GSM8K right, E2B → E4B | p | Time, E4B / E2B |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Repack | none | off | 1,189 → 1,226 | 0.00039 | 1.75 |
| Repack | yes | off | 1,188 → 1,223 | 0.00082 | 1.29 |
| Google | none | on | 1,194 → 1,235 | 0.000038 | 1.58 |
| Google | yes | on | 1,187 → 1,245 | 0.0000000019 | 1.39 |

All eight GSM8K pairs favour E4B beyond chance; the four above span the range, and the full list is in `derived.txt`. On AIME with thinking off, E4B got 10 to 13 right against E2B's 5 to 7, beyond chance in two of four setups; with thinking on the two models are level (9 to 12 against 6 to 10).

E4B generates at 0.52 to 0.68 times E2B's speed. Its GSM8K replies are 0.84 to 0.89 times as long, which brings the GSM8K time to 1.29 to 1.75 times E2B's. On AIME its replies are half to two thirds as long, so AIME runs took 0.74 to 1.47 times E2B's time.
The pattern matches E2B's. On GSM8K the answer shrinks by about a quarter and about 510 thinking tokens are added in front of it. With thinking off on AIME, E4B works the problem out in the answer itself; with thinking on, that working moves into the thinking text. E4B thinks less than E2B on both: 505 to 514 tokens against 601 to 615 on GSM8K, and 3,760 to 4,288 against 7,668 to 10,240 on AIME.
The drafter made every GSM8K run 1.99 to 2.39 times faster and every AIME run 1.38 to 2.23 times faster. The main model accepted 0.83 to 0.84 of the drafted tokens on GSM8K and 0.80 to 0.82 on AIME, and generation rose from 54.5–58.4 tokens per second to 121.6–132.3. That is a larger gain than on E2B, where the same drafter setup gave 1.55 to 2.04 times.
Thinking changed the answer on 54 to 59 GSM8K problems in each setup, and the gains outnumber the losses in all four, by 2 to 16. On E2B thinking changed 79 to 106 answers and the direction split both ways. On AIME no comparison does better than p = 0.2891. The full table of 24 tests is in `RESULTS.md`.
| | E2B | E4B |
| :--- | :--- | :--- |
| GSM8K score range | 0.8908 to 0.9052 | 🥇 0.9272 to 0.9439 |
| AIME right, thinking off | 5 to 7 | 🥇 10 to 13 |
| GSM8K time, drafter, thinking off | 🥇 0.77 to 0.84 h | 1.00 to 1.16 h |
| Drafter speed-up, GSM8K | 1.75 to 1.87 | 🥇 1.99 to 2.39 |
| Card memory, repack, no drafter | 🥇 3,610 MiB | 6,430 MiB |
For math on an 8 GB card, serve E4B with the drafter. It answers 35 to 58 more GSM8K problems than E2B, and with the drafter a full GSM8K pass takes 1.00 hour against E2B's 0.77.
- 🟢 E4B scored 0.9272 to 0.9439 on GSM8K, beyond chance above E2B in all eight setups.
