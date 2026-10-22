# Localization profiles

trajectories: 1629 (1629 solved, 0 unsolved); skipped (no trace or task row): 1439

`path` for a solved attempt is measured against its own verified patch (the place it really fixed); for an unsolved one against the reference fix.

| path | solved (own patch) | solved (reference fix) | unsolved (reference fix) |
| --- | --- | --- | --- |
| direct | 1521 | 1458 | 0 |
| searched | 48 | 39 | 0 |
| struggled | 60 | 113 | 0 |
| never_found | 0 | 19 | 0 |
| no_target | 0 | 0 | 0 |

solved somewhere else than the reference fix (no file in common): 51 of 1629

## solved, own patch

- calls before reaching it: median 0
- detour files before it: median 0, 4 or more: 6
- saw the exact changed lines: 1623 of 1629
- first edit outside it: 55

## By source (path as above)

| source | direct | searched | struggled | never_found |
| --- | --- | --- | --- | --- |
| r2e_gym | 261 | 4 | 1 | 0 |
| swe_gym | 507 | 25 | 32 | 0 |
| swe_smith | 301 | 3 | 3 | 0 |
| swebench | 452 | 16 | 24 | 0 |
