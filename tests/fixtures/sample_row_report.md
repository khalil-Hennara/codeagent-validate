# Stage 0 report

rows: 1  ok: 0  rejected: 1

## Output buckets (tasks/<bucket>.jsonl)


## Rejections (first error)

- E_PATCH_MALFORMED: 1

## All issue codes

- E_PATCH_MALFORMED: 1
- W_PATCH_HUNK_COUNTS: 1
- E_TEST_PATCH_MALFORMED: 1
- E_FIELD_MISALIGNED: 1
- W_PATCH_REPO_SUSPECT: 1
- E_NO_VERSION: 1
- E_PATCH_PATH_MISSING: 1
- E_TEST_PATCH_PATH_MISSING: 1

## By source

- swe_bench_verified: {'total': 1, 'rejected': 1}

## Sample rejected rows

### swe_bench_verified::test::23 (django__django-10554)
- [error] E_PATCH_MALFORMED: 5 hunk line(s) lack ' '/+/- prefix (whitespace stripped?) e.g. L5: '# the former is often more efficient'
- [warn] W_PATCH_HUNK_COUNTS: 1 hunk(s) with wrong line counts (git apply --recount may rescue)
- [error] E_TEST_PATCH_MALFORMED: 13 hunk line(s) lack ' '/+/- prefix e.g. L5: 'with pytest.raises(TypeError):'
- [error] E_FIELD_MISALIGNED: none of 2 FAIL_TO_PASS tests relate to test_patch (tests point at ['tests/queries/test_qs_combinators.py'], test_patch edits ['xarray/tests/test_dataarray.py'])
- [warn] W_PATCH_REPO_SUSPECT: gold patch top-level dirs ['xarray'] look unrelated to django/django
- [error] E_NO_VERSION: django/django uses SWE-bench per-version test specs but the row has no `version`; join it from the official dataset (scripts/export_official_meta.py + --official-meta)
- [error] E_PATCH_PATH_MISSING: gold patch modifies 2 file(s) absent at base_commit, e.g. ['xarray/core/dataset.py', 'xarray/core/variable.py'] (row text likely belongs to a different task)
- [error] E_TEST_PATCH_PATH_MISSING: test_patch modifies files absent at base_commit, e.g. ['xarray/tests/test_dataarray.py']
