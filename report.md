# Generation report

solved without recalled history: 596 of 2024 attempts (29%); the others are cleaned by `clean-recall`
attempts: 2024  solved: 1060  (52%)
tasks: 2024  solved: 1060  (52%; first try 52%)

| source | attempted | solved | rate | solved without recall |
| --- | --- | --- | --- | --- |
| r2e_gym | 316 | 198 | 63% | 128 |
| swe_gym | 623 | 331 | 53% | 213 |
| swe_smith | 200 | 159 | 80% | 99 |
| swebench | 885 | 372 | 42% | 156 |

## Outcomes

- SOLVED: 1060
- FAILS_F2P: 780
- MISMATCH_EXPECTED: 104
- NO_PATCH: 55
- BREAKS_P2P: 15
- NO_TEST_OUTPUT: 7
- TIMEOUT: 3

solved attempts whose reasoning recalls project history (to clean with `clean-recall`): 464 of 1060

unsolved with a patch: 909; half or more of F2P passing: 197; editing at least one file of the reference fix: 801

## Solve rate by size of the reference fix (changed source lines)

| lines | attempts | solved | rate |
| --- | --- | --- | --- |
| 1-10 | 619 | 412 | 67% |
| 11-30 | 656 | 325 | 50% |
| 31-80 | 494 | 195 | 39% |
| >80 | 255 | 128 | 50% |

## Cost and time (per attempt)

- prompt tokens: 272,865 (cached 91%)   completion tokens: 5,684
- longest context per attempt: median 13,841 tokens, max 74,834
- tool calls: 19.7   reasoning blocks captured: 11.3
- estimated cost: $0.232 per attempt, $0.442 per solved task, $0.787 per solve without recall
- wall time: 4.4 min per attempt
- projection for 2400 solved: ~4,583 attempts, ~$1,061.36

75 attempts hit infrastructure/API errors and will be retried on the next run.

accepted under the global limits (solved.jsonl): 1014 of 2400
- r2e_gym: 183
- swe_gym: 305
- swe_smith: 159
- swebench: 367
