# Evaluation integrity

This note records how the benchmark versions relate, how the held-out test
set was protected, and what was and was not allowed while tuning. The
README carries the numbers; this file carries the rules they were produced
under.

## Benchmark versions

| File | Built | Source | Sets | Status |
|---|---|---|---|---|
| `eval/manifest.json` (v1) | 2026-09-21 | GitHub ingest, newest-updated 6,000 issues | `dev` 100, `full` 500 of 650 eligible | Kept unchanged. 21 of its 100 dev cases have no issue text in the current corpus |
| `eval/manifest.v2.json` (v2) | 2026-10-10T23:18:29Z | The Postgres corpus (6,000 issues ingested 2026-10-10) | `test` 100, `dev` 100 of 632 eligible | Used for all tuning (dev) and the headline numbers (test) |

v2 uses the v1 eligibility rules and answer-key derivation without change;
both call the same function, `manifest.eligible_cases`. Postgres keeps no
record of whether GitHub truncated a list, so `miresolver benchmark` treats
a pull request with a full page of 100 files, or an issue with a full page
of 10 linked pull requests, as truncated. That can only reject a case v1
would have kept, never admit one it would have rejected. As a check,
replaying the v1 sampling (seed 20260921) over the rebuilt pool reproduces
the manifest that `miresolver ingest` wrote from the same window on
October 10 exactly, including every rejection count.

## How the test set was frozen

1. The eligible pool (632 cases) was rebuilt from the corpus.
2. Cases that appeared in any run made before v2 existed (the 100 v1 dev
   cases; the two smoke runs drew from the same set) were removed from the
   pool the test set is drawn from. 78 of them were still eligible. They
   may appear in dev, never in test.
3. The 100 test cases were drawn first with seed 20261010, then the 100
   dev cases from what remained. The two sets are disjoint.
4. `meta.test_sha256` is a SHA-256 over the canonical JSON of the test
   cases (`73c4101a...`). `miresolver evaluate --set test` refuses to run
   without `--final`, refuses `--limit`, and refuses a test set whose
   digest no longer matches.
5. The manifest was committed before any tuning run was started.

## Rules followed while tuning

- Every tuning decision was made on v2 dev runs only. Their results live
  under `eval/results/tuning/`.
- No case was added, dropped or reselected on the basis of any result.
- Answer keys and eligibility rules were not edited.
- The retrieval leakage filter was not loosened. It still excludes the
  case's own issue, anything fixed by the case's pull request or a later
  one, and issues with no known fix.
- Abstention thresholds were chosen by `miresolver calibrate` from a dev
  always-answer run, with the objective fixed in code before any run:
  maximise correct answers minus wrong answers, ties broken toward
  answering more cases and then toward lower thresholds. The command
  refuses a test-set run.
- The held-out test set was evaluated once per mode (abstaining and
  always-answer) with the final configuration, after tuning ended. No
  change was made after seeing those results.
- Every metric the runner produces is reported, including poor ones.

## Known weaknesses of this protocol

- With 100 test cases, one case moves a rate by one point and the
  standard error of a rate near 0.3 is about 4.6 points. Differences
  between tuning runs smaller than that are within noise.
- Model replies are not deterministic, so a rerun of the same
  configuration gives somewhat different numbers.
- 60 of the 100 test cases were also in v1's 500-case `full` set. That set
  was never run, so no results were seen for them.
- The leakage filter orders fixes by pull request number, a proxy for
  merge time.
