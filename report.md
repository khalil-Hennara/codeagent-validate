# Generation report

solved without recalled history: 936 of 2993 attempts (31%); the others are cleaned by `clean-recall`
attempts: 2993  solved: 1629  (54%)
tasks: 2959  solved: 1629  (55%; first try 55%)

| source | attempted | solved | rate | solved without recall |
| --- | --- | --- | --- | --- |
| r2e_gym | 421 | 266 | 63% | 170 |
| swe_gym | 1113 | 564 | 51% | 355 |
| swe_smith | 364 | 307 | 84% | 203 |
| swebench | 1095 | 492 | 45% | 208 |

## Outcomes

- SOLVED: 1629
- FAILS_F2P: 1124
- MISMATCH_EXPECTED: 141
- NO_PATCH: 69
- BREAKS_P2P: 19
- NO_TEST_OUTPUT: 8
- TIMEOUT: 3

solved attempts whose reasoning recalls project history (to clean with `clean-recall`): 693 of 1629

unsolved with a patch: 1295; half or more of F2P passing: 278; editing at least one file of the reference fix: 1157

## Solve rate by size of the reference fix (changed source lines)

| lines | attempts | solved | rate |
| --- | --- | --- | --- |
| 1-10 | 940 | 651 | 69% |
| 11-30 | 915 | 462 | 50% |
| 31-80 | 817 | 375 | 46% |
| >80 | 321 | 141 | 44% |

## Cost and time (per attempt)

- prompt tokens: 283,330 (cached 92%)   completion tokens: 5,824
- longest context per attempt: median 13,961 tokens, max 74,834
- tool calls: 20.2   reasoning blocks captured: 11.5
- estimated cost: $0.239 per attempt, $0.439 per solved task, $0.764 per solve without recall
- wall time: 4.2 min per attempt
- projection for 2400 solved: ~4,410 attempts, ~$1,053.59

75 attempts hit infrastructure/API errors and will be retried on the next run.

accepted under the global limits (solved.jsonl): 1577 of 2400
- r2e_gym: 246
- swe_gym: 538
- swe_smith: 307
- swebench: 486
