# Multi-Agent Issue Resolver

When a new issue arrives, this finds similar issues that were already solved,
investigates each candidate in parallel against the current code, and suggests
an approach with citations to the past issues and pull requests it relied on.
It abstains when nothing matches confidently.

**Status:** the issue corpus, ingestion CLI, provider-agnostic model layer, hybrid
retrieval (Qdrant dense plus keyword search), and the evaluation benchmark are
implemented and tested. The investigating and synthesizing agents are designed
(see `docs/specs`) but not built yet, so no accuracy results are reported.

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

Search runs dense vector similarity and keyword matching over the same query
and fuses the two rankings. Keyword matching matters here because issue text is
full of stack traces and symbol names, which embeddings blur and exact matching
catches.

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
