# Hybrid Retrieval Fix, Benchmark v2 and Abstention Calibration

**Goal:** Make keyword search contribute to retrieval, rebuild the benchmark so every case is scorable with a frozen held-out test set, tune only on dev, and report held-out numbers in an abstaining and an always-answer mode.

**Constraints:** answer keys, eligibility rules and the leakage filter do not change (the filter may only get stricter); no case is selected or dropped on the basis of results; the test set is created and committed before any tuning and is evaluated once per mode; total OpenAI spend under 10 dollars; the 500-case set is never run.

## Task 1: BM25 keyword search

- Files: `src/miresolver/retrieval/keyword.py` (new), `src/miresolver/retrieval/index.py`, `src/miresolver/retrieval/pipeline.py`, `src/miresolver/cli.py`, tests under `tests/retrieval/`.
- Root cause: `search_keyword` used a Qdrant `MatchText` filter, which requires every query token to be present, inside `scroll`, which does not rank. With a whole issue as the query only the issue itself matches.
- Fix: index a sparse term-frequency vector per document in a named sparse vector `bm25` with Qdrant's IDF modifier, so a sparse query scores documents by BM25. Tokenizer splits identifiers (camelCase, snake_case, dotted paths) and drops stopwords.
- Tests first: tokenizer behaviour; BM25 weights; a ranked keyword search over an in-memory collection returns a partially matching document; an old collection without the sparse vector fails with a message telling the user to re-index.
- Verify: retrieval-only check on live services, keyword hits per case well above one.

## Task 2: Benchmark v2

- Files: `src/miresolver/benchmark.py` (new), `src/miresolver/cli.py`, `src/miresolver/evaluation/runner.py`, tests.
- Rebuild the eligible pool from the Postgres corpus with `rules.is_eligible` unchanged. Postgres does not store GitHub's page-truncation flags, so a pull request with a full page of files, or an issue with a full page of linked pull requests, is treated as truncated (stricter, never looser). Check: replaying the v1 sampling over the rebuilt pool must reproduce `.cache/manifest.rebuilt.json` exactly.
- Split: draw the 100-case test set first with its own seed from the pool minus cases already run during earlier development, then the dev set from the remainder. Record seeds, creation time and a hash of the test case list in `meta`.
- `evaluate --set test` refuses to run without `--final`.
- Commit the manifest before any tuning run.

## Task 3: Modes, retrieval-only check, calibration

- `--mode always-answer` always runs synthesis and never abstains.
- `evaluate --retrieval-only` reports precedent availability without any chat call.
- Metric additions: hit@3 when a precedent is available, abstention rate when none is.
- Calibration: choose thresholds on the dev always-answer run by a rule fixed in advance (maximise correct answers minus confident wrong answers), then confirm on a dev abstaining run.

## Task 4: Final evaluation and documentation

- One held-out test run per mode with the final configuration, then README and `docs/evaluation-integrity.md`.
