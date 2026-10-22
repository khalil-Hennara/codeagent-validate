# codeagent_data

Turns the task file from the selection notebook into **validated, runnable tasks** for
generating teacher trajectories with the swegemma harness.

- **swegemma** supplies only the *agent side*: its tools and their output limits, the task
  prompt, nudges, context compaction and the trace format, so trajectories match what Gemma
  sees in the competition.
- **Each task's own Docker image** supplies the *environment*: Python version, dependencies,
  test command. Nothing from the competition's wheels or `setup.py` is used.

```
1.  selection notebook (v3) : picks ~10K tasks from 7 datasets   ─► sft_tasks_primary_10k.parquet
2.  ingest   : checks every row against its repo, no Docker       ─► tasks/*.jsonl
2b. select   : ranks tasks so the first N solved are the best mix  ─► ranked.jsonl, targets.json
3.  validate : runs the tests in each task's image, no model       ─► valid_tasks.jsonl
4.  generate : teacher (OpenRouter) + swegemma loop in each image,  ─► solved.jsonl + traces/
               then the task's tests on the teacher's patch
5.  parse    : localizer / fixer / scoring records                  (next)
```

`ingest` and `validate` run **one task per worker in parallel** and are **resumable** where it matters.
See [Speed](#speed) for sizing.

---

## 0. Install

Python 3.10+, `git`, GNU `patch`, and for step 3 (`validate`) a Linux x86 machine with Docker.

```bash
unzip codeagent_data.zip && cd codeagent_data
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # swebench==4.1.0, docker, pyarrow
python tests/test_stage0.py && python tests/test_stage1.py   # all lines should say "ok"
```

`swebench` is pinned to **4.1.0** on purpose: from 5.0 the per-repo test specs moved into the
Hugging Face rows. Your SWE-bench rows carry those scripts in `raw_json` anyway; 4.1 is the
fallback for rows that don't.

---

## 1. Produce the task file (notebook)

Run `swe_sft_data_selection_v3.ipynb` top to bottom. It writes
`sft_tasks_primary_10k.parquet` (+ `.jsonl`) and `sft_tasks_reserve.parquet`.

Columns this pipeline relies on: `instance_id, repo_canon, base_commit, problem_statement,
extra_context, patch, setup_patch, test_patch, f2p, p2p, env_image, verifier, version, raw_json`.
`raw_json` holds the original dataset row: SWE-bench rows carry the official `eval_script` and
`log_parser` there, and R2E-Gym rows carry `expected_output_json`. **Use the saved parquet,
not an in-memory export**, so `raw_json` is present.

---

## 2. Check every row: `ingest` (no Docker)

```bash
python -m codeagent_data ingest sft_tasks_primary_10k.parquet \
    --out out/stage0 \
    --repo-cache ~/.cache/repocache \
    --apply-check \
    --workers 16
```

| flag | meaning |
| --- | --- |
| `--repo-cache DIR` | blobless clone of each repo (commits + trees only; tens of MB each). Enables the checks against the real repository. Recommended. |
| `--apply-check` | apply setup patch → gold patch → test patch on the real files at `base_commit` (`git apply`, then `patch --fuzz=5` like the validator). |
| `--workers N` | rows checked in parallel (default 16). Network-bound: 16–32 is fine on a normal connection. |
| `--allow-eval-sources` | **train on SWE-bench Verified.** By default Verified rows go to `heldout_*` files so they stay a clean evaluation set for the paper. For the leaderboard alone you can include them. |
| `--allow-non-python` | keep Go/TypeScript/... rows. By default they are rejected (`E_LANG_UNSUPPORTED`) because the competition is Python-only. |
| `--limit N` | only the first N rows (quick try) |
| `--official-meta FILE` | optional; only needed for SWE-bench rows without `version` *and* without `eval_script` (see `scripts/export_official_meta.py`) |

**Outputs in `out/stage0/`:**

- `report.md`: counts, rejections by reason, sample rejected rows. **Read this first.**
- `tasks/<source>.jsonl`: rows that passed, one file per test-runner family (`swebench`,
  `swe_gym`, `swe_smith`, `r2e_gym`, `swe_bench_pro`, `promax`), plus `heldout_*` files.
  Each line is a swegemma-compatible task plus the fields the runners need.
- `normalized.jsonl`: every row with all its issues (including rejected rows).

What `ingest` fixes or changes on the way:

- **SWE-smith `base_commit`**: the dataset has none; it is taken from the instance id
  (`oauthlib__oauthlib.1fd52536.pr_879` → `1fd52536`).
- **Files created by a commit** (R2E-Gym's rebuilt diffs write them as edits, which can't
  apply) are re-marked as new files, using the real repository tree (`I_NEW_FILE_HEADERS_FIXED`).
- `hints_text` is emptied, because swegemma puts it in the agent prompt and hints often
  describe the fix. The original stays in `normalized.jsonl`.
- `extra_context` (SWE-bench Pro requirements/interface) is appended to the problem statement.

**Error codes (row is rejected):**

| code | meaning | what to do |
| --- | --- | --- |
| `E_PATCH_PATH_MISSING` / `E_TEST_PATCH_PATH_MISSING` | patch edits files that don't exist at `base_commit` | row text belongs to another task, or wrong commit: drop |
| `E_PATCH_APPLY` / `E_TEST_PATCH_APPLY` / `E_SETUP_PATCH_APPLY` | patch doesn't apply on the real files, even with fuzz | drop |
| `E_PATCH_MALFORMED` | hunk lines lost their ` `/`+`/`-` prefix | whitespace was stripped in an export step |
| `E_MISSING_FIELD` | a required field is empty | see the message |
| `E_NO_SETUP_PATCH` | SWE-smith row without its bug patch | file is from an old notebook version |
| `E_NO_EXPECTED_OUTPUT` | R2E-Gym row without `expected_output_json` | `raw_json` missing: use the saved parquet |
| `E_NO_VERSION` / `E_UNKNOWN_VERSION` | SWE-bench row with no eval_script and no usable version | use `--official-meta` |
| `E_NO_TESTS` | nothing to verify a fix with | drop |
| `E_NO_ENV` | no Docker image | drop (or build an environment) |
| `E_LANG_UNSUPPORTED` | not Python | keep with `--allow-non-python` |
| `E_DUPLICATE` | same task from two sources | the higher-priority copy is kept |
| `E_COMMIT_MISSING` | `base_commit` not in the repo | drop |

Warnings (`W_*`) keep the row. The main ones:

- `W_F2P_NOT_IN_TEST_PATCH` / `W_F2P_PARTIAL_LINK`: FAIL_TO_PASS names not found in test_patch.
  This is normal for Django (tests named by docstring) and mypy (tests in data files), so it
  isn't an error; step 2 is the real test.
- `W_GOLD_TOUCHES_PROTECTED`: the gold patch edits test/config files that the competition resets.
- `W_VALIDATE_UNSUPPORTED`: see the support table below.

---

## 2b. Rank tasks for the teacher: `select` (seconds, no Docker)

The research needs **N solved tasks** (default 2,400). The teacher won't solve every task, so
`select` doesn't pick N tasks. It builds **one ranked list**: the teacher stage walks down it
and stops once N tasks are solved. The full ~10K set can be used later for the leaderboard model.

```bash
python -m codeagent_data select --stage0 out/stage0 --out out/select --n 2400
cat out/select/selection_report.md
```

It works in three layers:

1. **Gates** (every task must pass all of them). Each one can be changed with
   `--gate name=value`, e.g. `--gate max_leak_ratio=0.6`.

   | gate | default | why |
   | --- | --- | --- |
   | not held out, accepted by `ingest`, Python | – | SWE-bench Verified stays the evaluation set |
   | no doubtful-label warnings | – | `W_GOLD_TOUCHES_PROTECTED`, `W_F2P_NOT_IN_TEST_PATCH`, `W_PATCH_REPO_SUSPECT`, `I_LARGE_PATCH`, `W_VALIDATE_UNSUPPORTED` |
   | `min_py_files`..`max_py_files` | 1..3 | Python source files the fix edits: localizable, fits the method |
   | `max_other_src_files` | 2 | docs/changelog next to the code change |
   | `min_changed_lines`..`max_changed_lines` | 2..150 | trivial and huge fixes make poor trajectories |
   | `min_f2p`..`max_f2p` | 1..30 | (R2E: `max_r2e_tests` 200) a sane number of tests to verify against |
   | `min_ps_chars`..`max_ps_chars` | 150..12,000 | a real description that leaves room in Gemma's 32k context |
   | `max_leak_ratio` | 0.3 | issue text that already contains the fix gives the location away |

2. **Order.** Sources are interleaved by weight, so **any prefix** of the list already has the
   intended mix; within a source, the best quality score comes first (+3 if the task has
   PASS_TO_PASS tests). Weights follow how trustworthy each source's tests are:

   | source | weight `--shares` | max share of the *solved* tasks `--max-share` | why |
   | --- | --- | --- | --- |
   | `swebench` (SWE-bench test/dev) | 0.40 | 50% | official per-repo harness, human-written issues and tests |
   | `swe_gym` | 0.35 | 50% | real issues and tests; generic pytest runner |
   | `r2e_gym` | 0.15 | 35% | real commits, LLM-written issues, hidden tests |
   | `swe_smith` | 0.10 | 20% | synthetic bugs, LLM-written issues |

   SWE-bench Pro (no runner yet) and ProMax (no FAIL_TO_PASS list) are left out unless you
   name them in `--shares`.

3. **Limits on solved tasks**, applied by the teacher stage while it runs: a task is skipped
   when its source already has its max share of the N solved tasks, or its repo already has
   `--max-per-repo` (default 120, 5%). When a source runs out of tasks, the others continue,
   so N is still reached if the pool is big enough.

**Outputs in `out/select/`:**

- `ranked.jsonl`: every task that passed the gates, in attempt order, with `selection_rank` and
  a cross-fitting `fold` (A/B, the plan's halves). Folds alternate within every (source, repo),
  so any prefix is balanced too.
- `targets.json`: N and the limits above; the teacher stage reads it.
- `selection_report.md`: what each gate removed, and per source how many solved tasks it can
  give if 40% / 55% / 70% of attempted tasks end up valid **and** solved. It warns if 2,400 is out
  of reach, and says roughly how many tasks will be attempted and how many images that pulls.

**Pilot before the full run.** The real valid-and-solved rate per source is unknown until the
teacher runs. Validate (and later run the teacher on) the first ~200 tasks of `ranked.jsonl`,
measure the rate per source, then check the report's table again:

```bash
head -n 200 out/select/ranked.jsonl > out/select/pilot.jsonl
python -m codeagent_data validate out/select/pilot.jsonl --out out/pilot --workers 8
```

For the full run, validation happens inside the teacher stage, task by task, so each image
is pulled once.

---

## 3. Prove each task works: `validate` (Docker, no model)

For every task, in fresh containers from its image:

1. **base run**: setup patch (SWE-smith only) + test patch → the tests that should be fixed must **fail**
2. **gold run** (×`--repeat`): setup patch + gold patch + test patch → they must **pass**

How "pass" is judged depends on what the row provides:

| mode | rows | rule |
| --- | --- | --- |
| `lists` | FAIL_TO_PASS / PASS_TO_PASS given | every F2P fails on base and passes with gold; every P2P passes with gold |
| `expected` | R2E-Gym | the gold run reproduces `expected_output_json` exactly (R2E's own reward rule); F2P = tests expected to pass that don't pass on base |
| `derive` | no F2P list (ProMax) | every test passes with gold, and at least one of them fails on base |

In `expected` and `derive` modes the FAIL_TO_PASS / PASS_TO_PASS lists it found are written
into `valid_tasks.jsonl`, so the teacher-verification step can use them.

**Smoke test first (≈10 tasks per source):**

```bash
for s in swebench swe_gym swe_smith r2e_gym promax; do
  python -m codeagent_data validate out/stage0/tasks/$s.jsonl --out out/smoke_$s --limit 10 --workers 4
done
cat out/smoke_*/report.md
```

Expect mostly `VALID`. If not, open `out/smoke_<source>/logs/<instance_id>.base.log` / `.gold0.log`.

**Full run on one machine:**

```bash
python -m codeagent_data validate out/stage0/tasks/{swebench,swe_gym,swe_smith,r2e_gym,promax}.jsonl \
    --out out/stage1 --workers 12
```

**Full run on several machines:** give each machine the same command with its own shard and
output folder. Shards never share an image, so each image is pulled on one machine only.

```bash
# machine 0 of 3                                         # machine 1 of 3 ...
python -m codeagent_data validate out/stage0/tasks/*.jsonl --shard 0/3 --out out/stage1_s0 --workers 12
```

Then combine them: `python -m codeagent_data merge --tasks out/stage0/tasks/*.jsonl --verdicts out/stage1_s*/verdicts.jsonl --out out/stage1`.

| flag | default | meaning |
| --- | --- | --- |
| `--workers` | 4 | tasks validated in parallel (see [Speed](#speed)) |
| `--shard K/N` | – | this machine takes shard K of N (0-based) |
| `--repeat` | 1 | gold runs per task; `2` detects flaky tests but doubles the cost |
| `--timeout` | 1800 | seconds per test run |
| `--mem` / `--cpus` | 8g / 2 | per container |
| `--offline` | off | disable network (some official scripts run `pip install -e .`, which can need it) |
| `--prune-images` | off | delete an image once the last task using it is done |
| `--no-pull` | off | only use images already on the machine |
| `--max-minutes` | – | stop starting new tasks after N minutes; re-run to continue |
| `--skip-done FILE…` | – | verdicts from earlier runs/machines: skip those tasks (failed pulls are retried) |

It is **resumable**: re-running the same command skips tasks already in `verdicts.jsonl`, except
`IMAGE_UNAVAILABLE` / `INFRA_ERROR` ones, which are retried.
Progress lines show an estimate of the time left.

**Outputs in `out/stage1/`:** `valid_tasks.jsonl` (input for rollouts), `verdicts.jsonl`,
`report.md`, `logs/`.

| verdict | meaning |
| --- | --- |
| `VALID` | keep |
| `F2P_PASSES_ON_BASE` | tests pass without the fix: task is trivial or broken |
| `GOLD_FAILS_F2P` / `GOLD_BREAKS_P2P` / `GOLD_FAILS_TESTS` | the reference fix doesn't pass in this image (or tests are flaky: see `flaky`) |
| `GOLD_MISMATCH_EXPECTED` | R2E-Gym: gold run differs from `expected_output_json` |
| `TEST_PATCH_APPLY_FAIL` / `GOLD_APPLY_FAIL` / `SETUP_APPLY_FAIL` | a patch doesn't apply in the image |
| `NO_TEST_OUTPUT` | the parser found no results: wrong test command or broken env |
| `IMAGE_UNAVAILABLE` | pull failed (wrong name, private, rate limit: `docker login`) |
| `TIMEOUT` | raise `--timeout` or drop |
| `UNSUPPORTED` | no runner for this source yet |

`strict_file_pass` in `verdicts.jsonl` records whether *every* test that ran passes with the
gold patch, which is the stricter rule swegemma applies at competition time.

---

## Where to run each step

| | `ingest` (no Docker) | `validate` (Docker) |
| --- | --- | --- |
| **Colab** | ✅ good: fast network, enough RAM | ❌ no Docker daemon in Colab |
| **Your laptop** (16 GB RAM, slow internet) | ✅ fine | only a smoke test: every task pulls a 0.3–3 GB image |
| **GitHub Actions, public repo** (recommended, free) | ✅ | ✅ 4 CPUs / 16 GB RAM per job, datacenter network, up to 20 jobs at once |
| Paid cloud VM with Docker | ✅ | ✅ simplest if you have credits |

### `ingest` on Colab

```python
!git clone <your fork of this repo> codeagent_data && cd codeagent_data && pip install -q -r requirements.txt
!cd codeagent_data && python -m codeagent_data ingest /content/drive/MyDrive/.../sft_tasks_primary_10k.parquet \
    --out out/stage0 --repo-cache /content/repocache --apply-check --workers 32
```

### `validate` on GitHub Actions

The workflow is in `.github/workflows/validate.yml`. Everything below is done from **Colab**
(where your files are) and the **GitHub website**; nothing needs to be downloaded to your laptop.

**A. On github.com (once)**

1. Create a new **public** repository, e.g. `codeagent-validate`, **empty** (no README, no
   .gitignore). Public gives 4 CPUs / 16 GB runners and free minutes; on a private repo the
   runners are 2 CPUs / 8 GB with 2,000 free minutes a month, which is enough for the pilot
   only. Job logs of a public repo are public; the task data comes from public datasets.
2. Create a token so Colab can push: avatar → *Settings* → *Developer settings* →
   *Personal access tokens* → *Fine-grained tokens* → *Generate new token*. Repository access:
   only `codeagent-validate`. Permissions → *Contents: Read and write*. **Also add
   *Workflows: Read and write***, otherwise GitHub refuses the push because it contains
   `.github/workflows/`. Copy the token.
3. Create a Docker Hub access token: hub.docker.com → *Account settings* → *Personal access
   tokens* → *Generate* (read-only is enough).
4. In the repository: *Settings* → *Secrets and variables* → *Actions* → *New repository
   secret*, twice: `DOCKERHUB_USERNAME` (your Docker Hub user name) and `DOCKERHUB_TOKEN`.

**B. In Colab: prepare the files and push**

```python
%cd /content/codeagent_data
# pilot: the first 200 ranked tasks (the list is interleaved, so they already mix every source)
!python scripts/make_task_files.py /content/out/select/ranked.jsonl --head 200 --out data/pilot
# full run (later, with Docker Pro): the first 4,500 ranked tasks
!python scripts/make_task_files.py /content/out/select/ranked.jsonl --head 4500 --out data/full
```

Adjust `/content/out/select/ranked.jsonl` to wherever `select` wrote it. Then push:

```python
from getpass import getpass
token = getpass("GitHub token: ")
user, repo = "YOUR_GITHUB_USER", "codeagent-validate"
!git init -q -b main && git add -A && git -c user.name="{user}" -c user.email="{user}@users.noreply.github.com" commit -qm "pipeline + task files"
!git push -q "https://{user}:{token}@github.com/{user}/{repo}.git" main
```

What gets pushed: the code (`codeagent_data/`, `scripts/`, `tests/`), `.github/workflows/`,
`requirements.txt`, and `data/pilot/` + `data/full/`. `.gitignore` keeps out `out/`, caches
and parquet files. Run `!git status --short | head` before committing if you want to check.

**C. Run the pilot (Actions tab)**

5. *Actions* → *validate-tasks* → *Run workflow*. The defaults are set for the pilot:
   tasks `data/pilot/*.jsonl`, 10 shards, 2 workers per job.
6. Watch the 10 `validate` jobs. When the `merge` job finishes, its **summary** shows the
   combined `report.md`; the `stage1` artifact (bottom of the run page) has
   `valid_tasks.jsonl`, `report.md`, `verdicts_merged.jsonl`. Each `shard-*` artifact has the
   logs of tasks that failed.
7. Send the report. Most tasks should be `VALID`. A whole source failing means its runner needs
   a fix before the full run. `IMAGE_UNAVAILABLE` on a free Docker account just means the pull
   limit was hit: those tasks are retried next time.

**Re-running after a runner fix.** Tasks with a final verdict are skipped on the next run. To run
only some of them again (e.g. `NO_TEST_OUTPUT` after a fix to the test scripts), download the
`stage1` artifact, copy its `verdicts_merged.jsonl` to `data/verdicts_done.jsonl` in Colab,
commit and push, then *Run workflow* with `retry` = `NO_TEST_OUTPUT` (several reasons:
space-separated). Everything else is skipped, so only those images are pulled again.
On the command line: `validate ... --skip-done data/verdicts_done.jsonl --retry NO_TEST_OUTPUT`.

**D. Full run (after subscribing to Docker Pro)**

8. *Run workflow* with tasks `data/full/*.jsonl`, `num_shards` 40. Up to 20 jobs run at once;
   each stops starting new tasks after `max_minutes` (jobs are killed at 6 h).
9. **To continue** (tasks left over, or failed pulls): download the `stage1` artifact, then in
   Colab copy its `verdicts_merged.jsonl` to `data/verdicts_done.jsonl`, commit, push (same push
   command), and run the workflow again with the same inputs. Finished tasks are skipped;
   `IMAGE_UNAVAILABLE` and `INFRA_ERROR` tasks are retried. The next `merge` includes the
   earlier verdicts, so the newest `stage1` artifact always has the full picture.

**Docker Hub's pull limit** is the real bottleneck, not CPU. Check yours (a check is not a pull):

```python
import requests
user, pat, image = "DOCKERHUB_USER", "DOCKERHUB_TOKEN", "ratelimitpreview/test"
tok = requests.get("https://auth.docker.io/token", auth=(user, pat),
                   params={"service": "registry.docker.io", "scope": f"repository:{image}:pull"}).json()["token"]
h = requests.head(f"https://registry-1.docker.io/v2/{image}/manifests/latest",
                  headers={"Authorization": f"Bearer {tok}"}).headers
print(h.get("ratelimit-limit"), h.get("ratelimit-remaining"))   # e.g. "200;w=3600": 200 pulls per hour
```

A pull is counted per image (not per layer), per account when logged in. Images already on the
machine cost nothing. When the limit is hit, the runner **waits instead of failing**: all workers
pause (1, 2, 4, 8, then every 10 minutes) and retry; only after `--rate-limit-wait` minutes
(default 90) does a task get `IMAGE_UNAVAILABLE`, which the next run retries.

At 200 pulls per hour, validating 4,500 tasks takes about a day of wall time whatever the number of
runners, so more parallel jobs don't help: use `num_shards` 10 and re-run until done (each run
stops cleanly after `max_minutes`). A Docker Pro subscription removes the cap and makes it a few hours.
SWE-bench Pro's images are on GitHub's registry and aren't affected.

**Validating before the teacher runs** costs a second pull of each image later (the teacher
stage needs the same images), but it means teacher calls are only spent on tasks whose tests
are known to work. With Docker Pro the extra pulls only cost time.

---

## Speed

Everything is parallel at the task level: one task per worker thread, no shared state
besides the output file. Rules of thumb for `validate`:

- **Workers ≈ CPU cores ÷ `--cpus`**, and at most RAM ÷ `--mem`. A 32-core / 128 GB machine:
  `--workers 14 --cpus 2 --mem 8g`.
- **Disk is usually the real limit.** Images are 0.3–3 GB each; tasks are sorted by image so
  each image is pulled once and its tasks run back to back. With little disk, add `--prune-images`.
- **More machines → `--shard`.** It scales linearly because no image is pulled twice.
- **`--repeat 1`** (default) halves the cost compared with flakiness checks; flaky tests also
  surface later, when teacher patches are verified.
- A SWE-bench task usually takes 1–5 minutes (two runs; some repos rebuild C extensions in
  their official script). Example: 7,800 tasks × ~3 min ÷ 14 workers ≈ 28 h on one machine,
  ≈ 9 h on three.

---

## Support by source

| source | test runner | status |
| --- | --- | --- |
| SWE-bench Verified / test / dev | official `eval_script` + `log_parser` from the row | ready |
| SWE-Gym | `pytest -rA <test_patch files>` (listed tests when the patch only edits data files, e.g. mypy) | ready, check the smoke test |
| SWE-smith | setup patch (the bug) first; `pytest -rA <files of F2P/P2P>` | ready, check the smoke test |
| R2E-Gym | the image's `run_tests.sh` + hidden `r2e_tests`, compared with `expected_output_json` | ready, check the smoke test |
| SWE-Bench ProMax | `pytest -rA <test_patch files>`, `derive` mode; repo assumed at `/testbed` | check the smoke test |
| SWE-bench Pro | per-task run scripts from `scaleapi/SWE-bench_Pro-os` | **not yet**: `UNSUPPORTED` |

---

## 4. Teacher trajectories: `generate` (Docker + an OpenRouter model)

For each task, in rank order, until the solved target is reached:

1. **Agent container** from the task's own image, **network off**. Before the agent starts
   (`generate/prepare.py`): the repo moves to `/workspace` (old path kept as a symlink so
   editable installs work); R2E's hidden tests and runner are deleted; every git ref that is not
   an ancestor of HEAD (upstream branches with the real fix, future tags, remotes, reflogs) is
   removed and pruned; for SWE-smith the bug (setup patch) is applied and folded into the base
   commit.
2. **swegemma's own agent loop** (`run_agent_sandbox`): same tools and output limits, same task
   prompt, nudges, budgets, `submit_patch`, stripping of test/config edits, same trace. Only
   swegemma's snapshot/wheels/`setup.py` preparation is switched off. Graph tools are not given.
   The teacher's instructions are in `teacher/` (`agent.yaml`, `prompts/system.md`).
3. **Verification** in a fresh container: setup patch + teacher patch + the task's tests, judged
   with the stage 3 lists: every FAIL_TO_PASS and PASS_TO_PASS test must pass → `SOLVED`.

Outputs per run folder: `results.jsonl` (one row per attempt: outcome, patch, tokens, cost
estimate), `traces/<id>.json` (raw entries: `thinking` = the teacher's reasoning, `text`,
`tool_call`, `tool_response`, `usage`), `logs/`, `generation_report.md`.
`generate-merge` combines shards and runs under the **global** per-source / per-repo limits
into `solved.jsonl` (task + `teacher: {patch, trace, stats, ...}`), the input of step 5.

The model is called through litellm's OpenRouter provider: OpenRouter's `reasoning` field
becomes `thinking` entries and is sent back on later turns. `tests/test_openrouter_wire.py`
checks this against a local fake OpenRouter server.

### Where to run it

| | |
| --- | --- |
| **GitHub Actions** (recommended) | ✅ the model runs at OpenRouter, so the runner only hosts containers; 20 jobs × 4 tasks in parallel |
| **Colab** | ❌ no Docker daemon, and every task needs its image |
| Your laptop / a VM with Docker | ✅ for a few tasks (same command as below) |

### The harness comes from the organizers' Kaggle dataset

The harness (`swegemma`, `adk_eval_core`, `adk_submission`) has no license file, so it is **not**
put in this public repository. The workflow downloads its wheels at run time from the organizers'
public dataset [metric/gemma-4-developer-agent-wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)
(`scripts/fetch_harness.py`, only the harness wheels, not vLLM/torch) and unpacks them with
`scripts/setup_harness.sh`. The log prints the google-adk / litellm versions found there, so
you can compare them with `requirements-generate.txt`.

1. kaggle.com → your profile → **Settings** → **API** → **Create New Token**. You get either a
   `kaggle.json` (`username` + `key`) or an API token string. The account must have accepted the
   competition rules.
2. In `codeagent-validate` → Settings → Secrets and variables → Actions → **Secrets**:
   `KAGGLE_USERNAME` + `KAGGLE_KEY` (from kaggle.json) **or** `KAGGLE_API_TOKEN`, and
   `OPENROUTER_API_KEY` (keep `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` from stage 3).

Alternative: a private repository with the harness zips, set as variable `HARNESS_REPO` with a
read-only token in secret `HARNESS_TOKEN`; when `HARNESS_REPO` is set it is used instead.

### Task files (from Colab)

```bash
cd /content/codeagent-validate
# valid tasks from the validate run (artifact `stage1`), in rank order, split under 40 MB
python scripts/make_task_files.py /content/stage1/valid_tasks.jsonl --head 100000 --out data/gen
cp /content/codeagent_data/out/select/targets.json data/targets.json    # solved target + caps
git add data/gen data/targets.json && git commit -m "generation tasks" && git push
```

### First test: 10 tasks

Actions → **generate-trajectories** → Run workflow with `num_shards=1`, `limit=10` (defaults),
and `prices` = the model's OpenRouter prices per 1M tokens `input output cached` (copy them from
the model page; they only feed the cost estimate). Optional: `reasoning_effort`, and
`extra_body` such as `{"provider": {"sort": "price"}}`.

The job summary shows the report. Check before scaling:
- **reasoning blocks captured** > 0 (otherwise the provider returns no reasoning),
- **cost per solved task** and the **projection** for 2,400,
- **longest context per attempt** (what the student must fit),
- the `generated` artifact: open a few `traces/*.json` and `solved.jsonl`.

### Full run

`num_shards=20`, `limit=0`. Each job stops starting tasks after `max_minutes` (default 290)
or when its share of the target is solved. To continue, just run the workflow again: nothing to
download or commit.

**Progress is kept on the `gen-ledger` branch** of this repository, written by the merge job of
every run (also after a cancel, a timeout or when the credit runs out, since each shard records
its attempts as it goes):

- `results.jsonl`: every attempt (task, try, outcome, tokens, cost estimate, patch, test progress)
- `traces/`: the trace of every solved attempt (permanent; artifacts expire after 90 days)
- `README.md` (counts), `report.md` (the latest generation report)

Each run reads the ledger first and **never starts a task that already has a finished attempt**,
solved or not. Only tasks whose attempt ended in an infrastructure or API error (the model did not
finish, e.g. `FATAL_API_ERROR: 402 Insufficient credits`) are tried again. A second try
(`ATTEMPTS`) happens only within the same run. Runs are queued one at a time (`concurrency`), so
they never write the ledger at once.

Final assembly, on Colab or locally:

```bash
git clone -b gen-ledger --single-branch https://github.com/<you>/codeagent-validate.git ledger
python -m codeagent_data generate-merge --tasks data/gen/*.jsonl --results ledger/results.jsonl --out out/final
python -m codeagent_data clean-recall --generated out/final --out out/cleaned
```

### Locally (any machine with Docker)

```bash
pip install -r requirements-generate.txt kaggle
python scripts/fetch_harness.py harness_src        # needs KAGGLE_USERNAME + KAGGLE_KEY (or KAGGLE_API_TOKEN)
bash scripts/setup_harness.sh harness_src harness && export PYTHONPATH=harness
export OPENROUTER_API_KEY=...
python -m codeagent_data generate data/gen/*.jsonl --targets data/targets.json --out out/gen \
    --limit 10 --workers 2 --price-in 0.6 --price-out 2.5 --price-cached 0.15
```

Useful flags: `--only <instance_id>` (one task), `--time-minutes`, `--tool-calls`, `--turns`,
`--model`, `--reasoning-effort`, `--extra-body`, `--prior` (results of earlier runs).

| outcome | meaning |
| --- | --- |
| `SOLVED` | all FAIL_TO_PASS + PASS_TO_PASS pass with the teacher's patch |
| `FAILS_F2P` / `BREAKS_P2P` / `MISMATCH_EXPECTED` (R2E) | the teacher's patch does not fix it / breaks other tests; the failing test names follow `failing:` and the test log is in `logs/<id>__a<N>.verify.log` |
| `NO_PATCH` | no edit at the end (budget or nudges exhausted) |
| `AGENT_PATCH_APPLY_FAIL` | the patch does not apply on a fresh container |
| `API_ERROR`, `INFRA_ERROR`, `IMAGE_UNAVAILABLE`, `PREPARE_FAILED` | retried on the next run |
| `FATAL_API_ERROR` | bad key / no credits: the job stops starting tasks |

How close an unsolved try came is recorded per attempt: `tests` (target tests still failing, share
of FAIL_TO_PASS passing) and `loc` (which files of the reference fix the teacher's patch edits).
When only 1-2 target tests fail, the tests are run once more before rejecting (stage 3 ran the
gold patch once, so a flaky test could reject a correct patch). A second try (`ATTEMPTS`) is only
spent on tasks where the first got somewhere: half or more of F2P passing, a file of the reference
fix edited, or no patch before the budget ran out. Only SOLVED trajectories go to `solved.jsonl`;
near-misses that edited the right files stay in `results_merged.jsonl` for localization-only use.
The report shows the solve rate by size of the reference fix; `MAX_GOLD_LINES` /
`MAX_GOLD_FILES` in the workflow (or `--max-gold-lines/--max-gold-files`) skip larger tasks.

The teacher must reason only from what it observes. Kimi nevertheless often "recalls" the
project's real fix in its reasoning (commit hashes, PR numbers, "pandas main today"), even when the
prompt forbids it; the student cannot do that. Every attempt counts such phrases
(`stats.recall_mentions`, 3 or more = recalls history). These solves are kept, flagged, and cleaned
after `generate-merge`: `clean-recall` rewrites only the affected reasoning entries (recalled
claims removed, the technical idea kept as the agent's own proposal, nothing added), checks the
result with the same detector, and drops an entry's text if two rewrites fail. Tool calls,
results and the patch are never changed; the original text stays in `cleaned_from` for audit.
It needs only internet and the OpenRouter key (Colab is fine):

```bash
pip install -r requirements.txt litellm==1.106.0
export OPENROUTER_API_KEY=...
python -m codeagent_data clean-recall --generated generated --out cleaned
```

Spot-check a few cleaned traces (`content` against `cleaned_from`) before training.
`generate-merge --drop-recall` leaves these solves out instead.

Before any container starts, `generate` sends one small request through exactly the agent's
model path (`preflight OK ... reasoning returned: yes` in the log); a wrong key, model id or a
missing dependency stops the job there. If the API still fails 3 times in a row during the run
(more with more workers), the job stops instead of running through the queue.

Tests: `PYTHONPATH=harness python3 tests/test_generate.py` (real agent loop, scripted model,
local sandbox), `tests/test_openrouter_wire.py` (real model path, fake server),
`tests/test_generate_docker.py` (real Docker; first `docker build -t toy/calc:1 tests/toy_image`).

### Localization profiles: `loc-stats`

How each trajectory found the code to change, measured against the reference fix (its source
files and hunk line ranges; docs and changelogs ignored): calls and searches before the first
read of a gold file, non-gold "detour" files opened before it, whether it looked at the exact
changed lines, whether the first edit went to a gold file. Each trajectory gets a `path`:
`direct`, `searched`, `struggled` (4+ detour files, 15+ calls before finding it, or a first edit
outside the fix) or `never_found`.

```bash
python -m codeagent_data loc-stats --tasks data/gen/*.jsonl --results ledger/results.jsonl \
    --trace-dirs shards/* --out out/loc        # shards/*: unzipped gen-shard artifacts (unsolved traces)
```

The ledger keeps only solved traces; unsolved ones (useful as "never found" / "struggled"
examples) live in the `gen-shard-*` artifacts, which GitHub deletes after 90 days.

## 5. Next: parse

y\*, L_edit, L_view, search queries and reasoning from `solved.jsonl` + traces, for the
localizer/fixer plan.

---

## Troubleshooting

- **`docker.errors.DockerException: Error while fetching server API version`**: Docker isn't
  running or your user can't reach `/var/run/docker.sock`.
- **Many `IMAGE_UNAVAILABLE`**: Docker Hub rate limit, so run `docker login`. For SWE-bench dev,
  images may not exist on the Hub.
- **Apple Silicon**: images are `linux/amd64`; run on an x86 Linux host.
- **Disk full**: `--prune-images`, or `docker system prune -a` between shards.
- **`ingest` slow**: raise `--workers`; the first run clones each repo once, later runs reuse
  `--repo-cache`.
