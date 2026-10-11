# Multi-Agent Issue Resolver

When a new issue arrives, this finds similar issues that were already solved,
investigates each candidate in parallel against the current code, and suggests
an approach with citations to the past issues and pull requests it relied on.
It abstains when nothing matches confidently.

**Status:** the issue corpus, ingestion CLI, provider-agnostic model layer
(embeddings and chat), hybrid retrieval (Qdrant dense vectors plus BM25
keyword ranking), the screening, investigating and synthesizing agents
orchestrated with LangGraph, and the evaluation runner are implemented and
tested. On a 100-case held-out test set (October 11, 2026, OpenAI models),
the system named a file the real fix changed in its top three for 40% of
issues when made to answer every case, and for 69% of issues that had a
similar solved issue to learn from. With abstention on, it answered 29% of
cases and was right on 66% of those, and only 10% of all cases got a
confident wrong answer. See "Evaluation results" below for every number and
its caveats.

## How results are measured

Accuracy is measured against real merged pull requests, so the answer key was
written by the maintainers who actually fixed each bug, not by us.

### Which issues count as evaluation cases

An issue is included only if all of the following hold:

- It carries a **bug label**
- It links to **exactly one merged** pull request. "Links" means GitHub's own
  closing relation, a pull request that says it closes the issue or was linked
  to it by hand. A pull request that merely mentions the issue number, such as
  a release changelog, is not a fix and does not count
- That pull request touches between **1 and 5** source files, which excludes
  mass reformatting and sweeping refactors
- The pull request is not a **revert**

An issue is also rejected when GitHub returned more linked pull requests, or
more changed files, than a single page held. A truncated list can hide a
second merged pull request and make an ineligible issue look eligible, so the
uncertain case is dropped rather than guessed at.

Rejected issues are counted by reason and reported alongside every run.

#### What counts as a source file

Only production source counts, both for the 1-to-5 rule and in the published
answer key. A one-line fix that also updates its test would otherwise put the
test file in the answer key, and a retrieval system naming that test would be
scored as correct. A path is dropped when any of these hold:

- A directory in the path is `test`, `tests`, `testing`, `__tests__`,
  `__mocks__`, `__snapshots__`, `spec`, `specs`, `fixtures`, `doc`, `docs`,
  `i18n`, `locale`, `locales`, or `translations`
- The filename contains `.test.`, `.spec.`, `_test.`, or `_spec.`, or starts
  with `test_` or `test-`
- The file ends in `.md`, `.rst`, `.txt`, `.po`, `.pot`, or `.snap`
- The file is a dependency lockfile: `cargo.lock`, `composer.lock`,
  `gemfile.lock`, `go.sum`, `npm-shrinkwrap.json`, `package-lock.json`,
  `packages.lock.json`, `pdm.lock`, `pnpm-lock.yaml`, `podfile.lock`,
  `poetry.lock`, `uv.lock`, or `yarn.lock`

Matching is case-insensitive. A pull request left with zero source files after
this filter is rejected as out of range.

### Leakage control

For each evaluation case the repository is indexed at the commit immediately
**before the fix** landed. Indexing the current repository would mean the fix is
already present in the code, which would inflate every number and make the
result meaningless.

Every case in the manifest carries a `base_commit`, and that is the commit to
index. It is the first parent of the pull request's merge commit. For a merge
commit or a squash merge, which is how the target repository lands nearly all
pull requests, that parent is the state of the base branch just before the fix
was applied.

One case is weaker and is worth knowing about: if a multi-commit pull request
was landed with a rebase merge, GitHub reports the last rebased commit as the
merge commit, and its first parent is an earlier commit of the same pull
request rather than the pre-fix base. Such a case would be indexed with part of
its own fix already present. This is not currently detected.

An issue whose fixing pull request has no merge commit that GitHub will report
is rejected as `missing_base_commit`, because there would be nothing to index.

### Reproducibility

The evaluation set is pinned in a committed manifest listing exact issue
numbers, the commit to index, and their answer keys. Selection is seeded, so
rebuilding from the same input produces identical `dev`, `full`, and `rejected`
sections. The `meta` block's `generated_at` records when the file was built and
therefore differs between runs, so the file is deterministic in its case data
rather than byte for byte.

The manifest carries a `meta` block so a reader holding only the file can say
what produced it: `repo`, `seed`, `dev_size`, `full_size`, `limit`,
`min_changed_files`, `max_changed_files`, `total_examined`, `total_eligible`,
`undersized`, and `generated_at`. Every number reported about a run is
recoverable from the committed file.

One honest caveat: issues are read newest-updated first, so the pool that
`--limit` draws from moves as the repository does. The manifest pins the cases
themselves, which is what the evaluation needs, but re-running ingestion later
examines a different window. `generated_at` records when the window was taken.

If fewer issues are eligible than `--full-size` asks for, ingestion fails
rather than quietly publishing a smaller set under the larger label. Pass
`--allow-undersized` to publish anyway; the manifest is then marked
`"undersized": true`.

## Setup

```bash
docker compose up -d postgres
pip install -e ".[dev]"
cp .env.example .env   # add your GitHub token
miresolver ingest --limit 2000 --out eval/manifest.json
```

Or, if you don't have Docker:

```bash
pip install -e ".[dev]"
cp .env.example .env   # add your GitHub token
miresolver ingest --limit 2000 --out eval/manifest.json --skip-db
```

`miresolver` reads `.env` from the working directory. Anything already
exported in the shell takes precedence over the file. `--skip-db` does not
need `DATABASE_URL`.

### Options for `miresolver ingest`

| Flag | Default | What it does |
|---|---|---|
| `--limit` | 2000 | How many closed issues to examine, newest-updated first |
| `--dev-size` | 100 | Size of the development set, a strict subset of the full set |
| `--full-size` | 500 | Size of the published set |
| `--seed` | 20260921 | Seeds the sampling. The same seed over the same eligible pool reproduces the manifest exactly, so record it with any number you publish |
| `--allow-undersized` | off | Publish even when fewer issues are eligible than `--full-size`, marked `"undersized": true` |
| `--skip-db` | off | Write only the manifest, never connect to Postgres |
| `--out` | `eval/manifest.json` | Where to write the manifest |

## Retrieval

Bring your own model. Copy `config/models.example.yaml` to `config/models.yaml`
and point each tier at whatever you run, including a private internal endpoint.
Any service speaking the OpenAI API works through `provider:
openai-compatible`; set `base_url` and the environment variable named by
`api_key_env`. Changing provider or model is a config edit, never a code
change.

```bash
docker compose up -d postgres qdrant
cp config/models.example.yaml config/models.yaml   # then edit it
miresolver index
miresolver search "kubelet crashes when the config file is empty"
```

Search runs dense vector similarity and BM25 keyword ranking over the same
query and fuses the two rankings by reciprocal rank. Keyword matching matters
here because issue text is full of stack traces and symbol names, which
embeddings blur and exact matching catches.

Keyword ranking is stored in Qdrant as a sparse vector per issue (named
`bm25`) holding saturated, length-normalised term frequencies; Qdrant's IDF
modifier supplies the inverse document frequency at query time, so the dot
product with the query's terms is the BM25 score. The tokenizer keeps whole
identifiers and also splits them, so `ErrInvalidConfigFormat` and
`file_watcher.ts` match both exactly and by their parts. A document does not
need to contain every query word, which matters because the query is a whole
bug report.

Up to the October 10 dev run, keyword search was a Qdrant full-text filter
(`MatchText`) inside an unranked `scroll`. `MatchText` requires every query
word to be present, so with a whole issue as the query the only match was the
issue itself, which the leakage filter then removed: retrieval was dense-only.
A collection built before the fix has no `bm25` vectors; searching it fails
with a message to re-run `miresolver index`, which replaces it.

Only symptom text is embedded: the issue title and body. The fixing PR, the
files it changed, and the commit to index against travel alongside as payload,
so a new symptom is matched against old symptoms rather than old solutions.

### Collection naming

`miresolver index` and `miresolver search` default `--collection` to a name
derived from the embed tier's configured model, e.g. `issues__text-embed-v1`,
rather than a bare `issues`. Vectors from two different embedding models are
not comparable, so switching models (which bring-your-own-model makes an
expected action, not an edge case) has to land in a new collection rather
than silently mixing with vectors from the old one in the same space. Both
commands derive the name the same way, so `search` always looks in the same
place `index` just wrote to, as long as `config/models.yaml`'s embed tier is
unchanged between the two runs.

If you switch models, the previous collection is left behind in Qdrant under
its old derived name; it is not touched or deleted automatically. Delete it
yourself once you no longer need it, or pass an explicit `--collection` to
either command if you want a name of your own choosing instead.

### Options for `miresolver index` and `miresolver search`

| Flag | Default | What it does |
|---|---|---|
| `--models` | `config/models.yaml` | Path to the model config file |
| `--qdrant-url` | `http://localhost:6333` | Qdrant server URL |
| `--collection` | derived from the embed model | Explicit override; see "Collection naming" above |
| `--batch-size` (`index` only) | 64 | How many documents to embed per request |
| `--limit` (`search` only) | 10 | Maximum number of ranked hits to return |

## How an issue is resolved

```
new issue (title, body) + the commit its code is at
        |
   retrieve: hybrid search, 20 candidate solved issues
        |
   screen: one batched call to the screen tier scores every candidate
        |
   top 3 candidates
        |
   investigate x 3, in parallel, each under its own deadline
        |
   synthesize: merge the verdicts, or abstain
        |
   ranked files + approach + citations + confidence
```

- **Screen.** A small model scores all candidates in one call, so cost does
  not grow with how many candidates retrieval returns. If screening fails,
  the candidates are kept in retrieval order, which degrades quality rather
  than losing the answer.
- **Investigate.** One agent per surviving candidate reads the files that the
  candidate's fix changed, as they stood at the new issue's base commit, and
  returns a verdict: does the precedent apply, which files are implicated
  now, what approach it suggests, and a confidence. An investigator that
  fails or runs past its deadline becomes a failed verdict; the others are
  still used.
- **Synthesize.** A stronger model merges the verdicts into one ranked file
  list and approach. Citations are limited to issues that were actually
  investigated. Abstention is decided by fixed rules, not by the model
  alone. The system answers only when all three gates pass: some
  investigator found an applicable precedent with confidence at or above
  the verdict gate (default 0.85), the synthesizer's own confidence is at
  or above the synthesis gate (default 0.9), and at least two investigators
  (`--min-agreement`) named one of the top three suggested files. The
  defaults were chosen by `miresolver calibrate` on the dev set; see
  "Evaluation results". With `--mode always-answer` the gates are skipped
  and every case gets an answer. If the synthesis call fails, the answer is
  merged from the verdicts directly.

Code is read at the pinned commit, never the current tree, either file by
file from `raw.githubusercontent.com` (cached under `.cache/code`, so a
repeat run makes no requests) or from a local clone with `--local-repo`.
Fetching single files avoids cloning microsoft/vscode, about 1GB, and
checking out one commit per case.

### Why LangGraph rather than the Go coordinator in the design

The original design puts fan-out, timeouts and cancellation in a Go service
calling Python workers over gRPC. The workflow is built as a LangGraph state
graph in Python instead: its `Send` primitive is the same map-reduce shape
(one task per candidate, run concurrently, verdicts collected by a reducer),
and per-investigator deadlines with partial results are enforced inside the
investigator node. Building a second runtime and an RPC boundary before a
single answer had been measured would have put the riskiest infrastructure
ahead of the evidence that decides whether the product is worth having. The
node boundaries are the ones a Go coordinator would call, so that move
remains open. The reasoning is in
`docs/specs/2026-10-10-agents-and-evaluation-design.md`.

### Model providers for the agent tiers

| Provider | Status |
|---|---|
| `openai-compatible` | Implemented and tested against the documented wire format. The configuration used for evaluation |
| `anthropic` | Native Messages API adapter, implemented and tested against the documented wire format with a mocked transport; not yet exercised against the live API |
| `gemini` | Native `generateContent` adapter, implemented and tested the same way; not yet exercised against the live API |

The embed tier must be `openai-compatible`; the native adapters refuse it
loudly.

## Resolving an issue

```bash
cp config/models.openai.example.yaml config/models.yaml   # or your own
miresolver resolve "Terminal hangs on resize" --body "Steps: ..." \
    --base-commit 6629039adedf1b0009b43d0c4f2f8d8cd52e2bc2
miresolver resolve --issue 148034 --base-commit 6629039adedf1b0009b43d0c4f2f8d8cd52e2bc2
```

The output lists the suggested files, most likely first, the approach, and
the past issues and pull requests it is based on, or says "No confident
match" with the precedents that were judged insufficient. An abstention
never prints files.

### Options for `miresolver resolve`

| Flag | Default | What it does |
|---|---|---|
| `title` (positional) | none | Title of the new issue |
| `--body`, `--body-file` | empty | Body text, inline or from a file |
| `--issue` | none | Read title and body of this issue from the Postgres corpus instead |
| `--base-commit` | required | Commit whose code the investigators read |
| `--json` | off | Print the full result, including every verdict and token counts, as JSON |

### Options shared by `resolve` and `evaluate`

| Flag | Default | What it does |
|---|---|---|
| `--models` | `config/models.yaml` | Model config |
| `--qdrant-url`, `--collection` | as for `search` | Where the index lives |
| `--repo` | `TARGET_REPO`, else `microsoft/vscode` | Repository whose code is read |
| `--local-repo` | none | Read code from this local clone with git instead of fetching it |
| `--code-cache` | `.cache/code` | Cache for fetched files |
| `--retrieve-k` | 20 | Candidates retrieved |
| `--investigators` | 3 | Candidates that get an investigator |
| `--abstain-threshold` | 0.85 | Verdict gate: abstain unless some investigator's applicable verdict reaches this confidence |
| `--synthesis-threshold` | 0.9 | Synthesis gate: abstain unless the synthesizer's confidence reaches this |
| `--min-agreement` | 2 | Abstain unless this many investigators named one of the top three files |
| `--mode` | `abstain` | `always-answer` skips the three gates and answers every case |
| `--retrieval` | `hybrid` | `dense` or `keyword` to use one ranking alone |
| `--investigator-timeout` | 90 | Seconds before an investigator is cut off |

## Evaluation

```bash
docker compose up -d postgres qdrant
cp .env.example .env            # add GITHUB_TOKEN and OPENAI_API_KEY
cp config/models.openai.example.yaml config/models.yaml
miresolver ingest --limit 6000 --out .cache/manifest.rebuilt.json   # fills Postgres
miresolver index
M=eval/manifest.v2.json
miresolver evaluate --manifest $M --estimate   # upper-bound cost, no model calls
miresolver evaluate --manifest $M --set dev --max-cost 3
miresolver evaluate --manifest $M --set dev --mode always-answer --tag t1-always-answer
miresolver calibrate eval/results/dev-n100-seed20260921-...-t1-always-answer
miresolver evaluate --manifest $M --set test --final --mode always-answer --tag always-answer
miresolver evaluate --manifest $M --set test --final --tag abstain
```

`--manifest` still defaults to the v1 file, `eval/manifest.json`, so pass
the v2 benchmark explicitly.

### Benchmark v2 and the held-out test set

`eval/manifest.v2.json` holds two disjoint 100-case sets drawn from the
stored corpus: `dev`, used for every tuning decision, and `test`, held out
until tuning ended and then run once per mode. How the test set was frozen
and the rules followed while tuning are in `docs/evaluation-integrity.md`.

```bash
miresolver benchmark --exclude-run eval/results/dev-n100-seed20260921-20261010T221905Z
```

| Flag | Default | What it does |
|---|---|---|
| `--out` | `eval/manifest.v2.json` | Where to write the manifest |
| `--seed` | 20261010 | Seeds the test draw, then the dev draw |
| `--test-size` | 100 | Size of the held-out test set |
| `--dev-size` | 100 | Size of the dev set |
| `--exclude-run` | none | Results directory of an earlier run, repeatable; its cases may enter dev but never test |

`miresolver evaluate --set test` refuses to run without `--final`, refuses
`--limit`, and refuses a test set whose SHA-256 digest no longer matches
`meta.test_sha256`.

### Calibrating abstention

`miresolver calibrate RUN` reads a dev always-answer run, where every case
was answered and the abstaining rule's decision was recorded alongside, and
searches verdict gate, synthesis gate and minimum agreement for the setting
that maximises correct answers minus wrong answers, ties broken toward
answering more cases and then toward lower thresholds. It writes
`calibration.json` into the run directory and refuses a test-set run.

Ingestion examines the newest-updated issues, so the corpus it stores drifts
as the repository does. Writing the rebuilt manifest aside keeps the
committed one, and therefore the cases, fixed; each case's text is read from the corpus, and a case whose issue is no
longer in the examined window is recorded as `skipped_no_text` rather than
scored.

### What is measured

Predicted files are compared with the answer key, the production source files
the fixing pull request changed.

| Metric | Meaning |
|---|---|
| hit@1, hit@3 | How often a correct file was ranked first, or in the top 3 |
| precision@3, recall@3 | Of the top 3 suggested files, the share that were correct; of the correct files, the share that appeared in the top 3 |
| abstention rate | How often the system said "no confident match" |
| accuracy when answering | hit@3 over the cases where it did not abstain |
| confident-but-wrong rate | Share of all cases where it answered and missed, the failure that matters most |
| precedent available | Share of cases where some retrieved past fix touched a correct file, so a right answer was reachable |
| correct abstention rate | Of the abstentions, the share where no such precedent existed |
| latency | Median and 95th percentile seconds per case |
| cost | Tokens per tier, and dollars where the config sets prices |

File metrics are reported over all cases, using what the system would have
suggested even when it abstained, and separately over answered cases only, so
abstaining cannot hide misses or inflate accuracy.

### Leakage control in retrieval

The index holds every corpus issue, including the evaluation cases, so for
each case retrieval excludes the case's own issue, anything fixed by the
case's own pull request, anything fixed by a pull request numbered at or
above it, and issues with no known fix. Pull request numbers are assigned when
a pull request is opened, so this approximates "fixed before this fix
existed" and errs toward excluding.

### Reproducibility and cost guardrails

- `--limit n` draws a seeded sample from the chosen set (`--seed`, default
  20260921), so the same seed names the same cases. Model replies are not
  deterministic; the seed fixes which cases run, not what the models say.
- `--set full` is refused without `--allow-full`.
- `--max-cost` (default 3 dollars) stops the run once priced spend reaches
  it and records `stopped_early: true`. Recorded spend is a floor: query
  embeddings, cases that errored, and investigators that timed out are not
  metered, and the summary counts the last two under `spend.unmetered`.
- `--estimate` prints an upper bound from the prompt budgets and configured
  prices without calling any model. With `config/models.openai.example.yaml`
  and the default options, a 100-case v2 set estimates at about 3.43
  dollars, before indexing. Actual metered spend has been under half of
  that.

### Options for `miresolver evaluate`

| Flag | Default | What it does |
|---|---|---|
| `--set` | `dev` | `dev` (100 cases), `test` (100, v2 only) or `full` (500, v1 only) |
| `--final` | off | Required for `--set test`, the one held-out run per mode |
| `--retrieval-only` | off | Score retrieval alone, the share of cases with a usable precedent at depths 1 to 40; no chat calls |
| `--tag` | none | Suffix for the run directory name |
| `--limit` | all | Seeded subset size |
| `--seed` | 20260921 | Seeds the subset |
| `--allow-full` | off | Required to run the full set |
| `--max-cost` | 3 | Stop once priced spend reaches this many dollars |
| `--estimate` | off | Print the cost upper bound and exit |
| `--manifest` | `eval/manifest.json` | Case list and answer key |
| `--out-root` | `eval/results` | Where run directories are written |

Each run writes `eval/results/{set}-n{count}-seed{seed}-{UTC time}[-{tag}]/` with
`summary.json` (metrics, model per tier, case counts, tokens, spend, options,
the estimate) and `cases.jsonl` (one line per case: predictions, citations,
verdicts, confidence, abstention, latency, tokens). Every reported number
comes from these files.

### Evaluation results

All numbers below are copied from `summary.json` in the run directories
named, and the per-case detail is in `cases.jsonl` beside each one. The
headline numbers come from the held-out test set of benchmark v2, run once
per mode with the final configuration after tuning ended. Dev numbers are
tuning runs and are shown only to explain the choices.

Models: `text-embedding-3-small` (embed), `gpt-4.1-mini` (screen and
investigate), `gpt-4.1` (synthesize), through
`config/models.openai.example.yaml`. Options: 20 candidates retrieved, 3
investigated, hybrid retrieval, verdict gate 0.85, synthesis gate 0.9,
minimum agreement 2.

#### Held-out test set (100 cases, October 11, 2026)

Runs: `eval/results/test-n100-seed20260921-20261011T190246Z-always-answer`
and `eval/results/test-n100-seed20260921-20261011T202023Z-abstain`. Both
scored all 100 cases with 0 errors, 0 skipped and no early stop.

| Metric | Always-answer | Abstaining |
|---|---|---|
| Cases answered | 100 | 29 |
| hit@1, all cases | 0.30 | 0.29 |
| hit@3, all cases | 0.40 | 0.39 |
| precision@3, all cases | 0.180 | 0.172 |
| recall@3, all cases | 0.268 | 0.265 |
| Accuracy when answering (hit@3 of answered cases) | 0.40 | 0.655 |
| hit@1 of answered cases | 0.30 | 0.448 |
| Confident-but-wrong rate (share of all cases) | 0.60 | 0.10 |
| Correct abstention rate | n/a | 0.634 |
| hit@3 with a usable precedent retrieved (49 cases) | 0.694 | 0.633 |
| hit@3 without one (51 cases) | 0.118 | 0.157 |
| Latency per case, median / p95 | 8.4 s / 10.6 s | 7.7 s / 9.6 s |
| Metered spend | 1.52 dollars | 1.37 dollars |

hit@k counts a case as solved when any of the top k suggested files is one
the real fix changed. In the abstaining run, the "all cases" rows score the
ranked files the system produced even when it then abstained.

What the numbers say:

- The system is only as good as its precedents. When retrieval surfaced a
  solved issue that touched the right code, which happened in 49% of
  cases, the top three files contained a correct one about 69% of the
  time. Without one it rarely found the file (12%), so most misses are
  retrieval and corpus coverage, not investigation.
- Abstention trades coverage for trust. With the gates on, the system
  answers about three cases in ten, is right on two thirds of them, and
  gives a confident wrong answer on one case in ten, against six in ten
  when it must always answer. It still abstains on about half of the
  cases that did have a usable precedent, so the gates are conservative.

#### How the configuration was chosen (dev set, tuning only)

| Dev run (`eval/results/tuning/`) | Answered | hit@3, all | Accuracy when answering | Confident-but-wrong | Spend |
|---|---|---|---|---|---|
| Starting point: 10 candidates, `gpt-4o-mini` screen, one 0.5 gate | 66 | 0.25 | 0.24 | 0.50 | 0.90 |
| `t1-always-answer`: 20 candidates, fix paths shown to the screener, `gpt-4.1-mini` screen | 100 | 0.34 | 0.34 | 0.66 | 1.51 |
| `t2-abstain-calibrated`: t1 plus the calibrated gates (93 scored, 7 errors) | 18 | 0.31 | 0.50 | 0.10 | 1.24 |

Retrieval alone, on the 100 dev cases, the share with a usable precedent
in the top k (`retrieval-dev-*` runs):

| Retrieval | k=1 | k=3 | k=10 | k=20 | k=40 |
|---|---|---|---|---|---|
| Hybrid | 0.18 | 0.26 | 0.36 | 0.46 | 0.52 |
| Dense only | 0.11 | 0.21 | 0.39 | 0.45 | 0.52 |
| Keyword only | 0.15 | 0.22 | 0.34 | 0.41 | 0.48 |

Hybrid ranks the best precedent first more often than either ranking
alone, and widening from 10 to 20 candidates raises precedent coverage
from 36% to 46%, which is why 20 became the default. `miresolver calibrate`
on the t1 run chose the gates: at 0.85, 0.9 and agreement 2 it answered 25
of 100 dev cases with 14 right and 11 wrong, against 26 right and 46 wrong
at the old 0.5 setting. The 7 errors in t2 were Qdrant request timeouts,
which led to the 60-second Qdrant timeout; that change does not affect
scoring.

#### Earlier results

The first dev run (October 10, `eval/results/dev-n100-seed20260921-20261010T221905Z`)
used benchmark v1, scored 79 of 100 cases, and reached hit@3 0.28 with
keyword search effectively disabled by the bug described under "Retrieval".
It is kept for the record and is not comparable with the v2 numbers.

Caveats:

- Each mode was run once on the test set. Model replies are not
  deterministic, so a rerun gives somewhat different numbers, and with 100
  cases one case moves a rate by one point.
- Two test runs did not complete and were discarded as incomplete: the
  first always-answer run lost network access (78 of 100 cases failed to
  embed), and the first abstaining run stopped making progress after 60
  cases and was killed after 40 minutes; the cause was not confirmed. Each
  was rerun once from scratch with the same configuration. Details are in
  `docs/evaluation-integrity.md`.
- The retrieval leakage filter orders fixes by pull request number, a proxy
  for merge time, so a precedent fixed by an earlier-numbered pull request
  that merged later can still be retrieved.
- Recorded spend is a floor. Query embeddings and the index embeddings are
  not metered by the runner.

### Known limitations

- The held-out test set has 100 cases and was run once per mode; the
  500-case v1 full set has not been run.
- The native Anthropic and Gemini chat adapters are tested only against the
  documented wire format with a mocked transport.
- Not yet built from the original design: the code-search-only baseline arm
  of the central experiment, the single-agent versus fan-out comparison,
  the Redis cache, Prometheus metrics and SLO alerts, and the hand-graded
  sample of suggested approaches.
- The retrieval leakage filter orders fixes by pull request number, a proxy
  for merge time.
- Cases whose issue text is outside the ingested window are skipped and
  counted, never scored.
- Error messages from a failed model call include the endpoint URL, so a
  `base_url` that embeds credentials would expose them in `cases.jsonl`.
  Pass keys through `api_key_env`.
