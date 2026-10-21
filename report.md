# Generation report

solved without recalled history: 41 of 154 attempts (27%); the others are cleaned by `clean-recall`
attempts: 154  solved: 81  (53%)
tasks: 154  solved: 81  (53%; first try 53%)

| source | attempted | solved | rate | solved without recall |
| --- | --- | --- | --- | --- |
| r2e_gym | 32 | 15 | 47% | 7 |
| swe_gym | 46 | 25 | 54% | 15 |
| swe_smith | 14 | 11 | 79% | 6 |
| swebench | 62 | 30 | 48% | 13 |

## Outcomes

- SOLVED: 81
- FAILS_F2P: 55
- MISMATCH_EXPECTED: 16
- BREAKS_P2P: 1
- NO_PATCH: 1

solved attempts whose reasoning recalls project history (to clean with `clean-recall`): 40 of 81

unsolved with a patch: 72; half or more of F2P passing: 20; editing at least one file of the reference fix: 62

## Solve rate by size of the reference fix (changed source lines)

| lines | attempts | solved | rate |
| --- | --- | --- | --- |
| 1-10 | 24 | 16 | 67% |
| 11-30 | 52 | 28 | 54% |
| 31-80 | 48 | 18 | 38% |
| >80 | 30 | 19 | 63% |

## Cost and time (per attempt)

- prompt tokens: 305,560 (cached 93%)   completion tokens: 6,501
- longest context per attempt: median 14,526 tokens, max 69,481
- tool calls: 19.9   reasoning blocks captured: 11.3
- estimated cost: $0.263 per attempt, $0.500 per solved task, $0.988 per solve without recall
- wall time: 4.1 min per attempt
- projection for 2400 solved: ~4,563 attempts, ~$1,200.22

74 attempts hit infrastructure/API errors and will be retried on the next run.

accepted under the global limits (solved.jsonl): 81 of 2400
- r2e_gym: 15
- swe_gym: 25
- swe_smith: 11
- swebench: 30
