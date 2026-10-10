# Agents and Evaluation: Design

**Date:** 2026-10-10
**Status:** Approved, implemented by `docs/plans/2026-10-10-chat-models-agents-and-evaluation.md`
**Parent spec:** `docs/specs/2026-09-21-multi-agent-issue-resolver-design.md`

This document fills in the part of the parent design that was never planned: chat-completion support in the model layer, the screening, investigation and synthesis agents, and the evaluation runner that scores them against the committed answer key. Where it departs from the parent spec, the departure and its reason are stated.

## Scope

In scope:

1. A `ChatClient` protocol with three adapters: OpenAI-compatible, native Anthropic Messages, native Gemini `generateContent`.
2. The resolution workflow: retrieve, screen, investigate in parallel, synthesize or abstain.
3. Read access to repository source at an arbitrary pinned commit.
4. An evaluation runner that writes seeded, reproducible results under `eval/results/`.
5. CLI subcommands `resolve` and `evaluate`.

Out of scope for this iteration, unchanged in the parent spec: the Go coordinator, gRPC, Redis caching, Prometheus metrics and SLO alerting, the baseline (code-search-only) arm of the central experiment, and the hand-graded approach sample.

## Departure from the parent spec: orchestration in Python with LangGraph

The parent spec puts fan-out, timeouts and cancellation in a Go coordinator that calls Python workers over gRPC. This iteration builds the workflow as a LangGraph `StateGraph` in the same Python process instead.

Reasons:

- No agent existed yet. Building a second language runtime and an RPC boundary before there is a single measured answer front-loads the riskiest infrastructure ahead of the evidence that decides whether the product is worth having (the central experiment).
- LangGraph's `Send` primitive is exactly the map-reduce shape the design describes: one node emits one task per screened candidate, the tasks run concurrently, and a reducer collects their verdicts for the synthesizer.
- Per-investigator deadlines and partial results are still enforced, inside the investigator node with `asyncio.wait_for`. A timed-out or failed investigator yields a verdict marked `failed` rather than failing the run, so the synthesizer always receives every slot and the evaluation can count failures.

Tradeoff accepted: the concurrency engineering the parent spec wanted to showcase in Go is reduced to Python asyncio. The graph's node boundaries (retrieve, screen, investigate, synthesize) are the same boundaries a Go coordinator would call, so moving fan-out into Go later replaces the orchestration layer without rewriting the agents.

## Model layer: chat tiers

`ChatClient.complete(messages, *, max_tokens, json_mode) -> ChatResult` where `ChatResult` carries the text and a `Usage(input_tokens, output_tokens)`. The system prompt is passed as a separate argument because Anthropic and Gemini both take it outside the message list.

| Adapter | Endpoint | Notes |
|---|---|---|
| `openai-compatible` | `POST {base_url}/chat/completions` | `json_mode` sends `response_format: {type: json_object}`. Covers OpenAI, Azure, vLLM, Ollama, Groq, Together, internal gateways |
| `anthropic` | `POST {base_url or https://api.anthropic.com}/v1/messages` | Native. `json_mode` is enforced by prompt instruction plus tolerant parsing, since the Messages API has no generic JSON switch |
| `gemini` | `POST {base_url or https://generativelanguage.googleapis.com}/v1beta/models/{model}:generateContent` | Native. `json_mode` sets `responseMimeType: application/json` |

All three are plain `httpx` against the documented wire format, tested with `httpx.MockTransport`. None has been exercised against a live endpoint in this iteration; the README says so.

Tier config gains two optional fields, `input_cost_per_mtok` and `output_cost_per_mtok` (US dollars per million tokens). They are used only to report and cap spend. A tier without prices reports token counts and a cost of `null`, never an invented figure.

## Repository access at a pinned commit

The investigator must read code as it stood at the case's `base_commit`, never the current tree, or the fix leaks into the evidence.

`CodeReader.read(commit, path) -> str | None` with two implementations:

- `GitHubRawCodeReader` fetches `https://raw.githubusercontent.com/{repo}/{commit}/{path}` and caches each file on disk under `{cache_dir}/{commit}/{path}`. This avoids cloning microsoft/vscode (about 1GB) and fetching one commit per case; an investigator needs a handful of files, not a tree. A 404 returns `None` (the file did not exist at that commit), which is itself evidence the investigator is told about.
- `LocalGitCodeReader` runs `git show {commit}:{path}` in a local clone, for offline or private repositories.

Each file is truncated to a configurable character budget before it reaches a prompt.

## Workflow

```
new issue (title, body) + base commit
        |
   retrieve: hybrid search, k=10, minus excluded issues
        |
   screen: one batched call to the screen tier, scores 0..1 per candidate
        |
   top 3 candidates with score >= screen floor
        |
   +----+----+----+          (LangGraph Send, concurrent, per-agent timeout)
   investigate x N
   +----+----+----+
        |
   synthesize: synthesize tier merges verdicts, then deterministic rules
        |
   Resolution: approach, ranked files, citations, confidence, abstained
```

**Retrieve.** Uses the existing `pipeline.search`. A caller-supplied exclusion predicate removes candidates; the evaluation uses it for leakage control (below).

**Screen.** One call lists every candidate's number, title and the first 600 characters of its body against the new issue, and asks for `{"scores": [{"issue": n, "score": s}]}`. Candidates the model omits score 0. If the call or its parsing fails, screening falls back to retrieval order, so a screening outage degrades quality rather than availability. Screening never names files.

**Investigate.** One agent per screened candidate. Its prompt holds the new issue, the past issue, the past fix PR number and its changed files, and the current content of those files at the new issue's base commit (or a note that a file does not exist there). It returns `{"applies": bool, "confidence": 0..1, "files": [...], "approach": str, "reasoning": str}`. Files outside the precedent's changed files are allowed, because a fix can move, but each must be a repository path.

**Synthesize.** The synthesize tier receives every successful verdict and returns `{"approach", "files": [...ranked], "cited_issues": [...], "confidence"}`. Deterministic rules then apply, so abstention never depends on the model alone:

- Citations are filtered to issues that were actually investigated; each citation carries the issue and its PR.
- If no verdict has `applies` true with confidence at or above `abstain_threshold` (default 0.5), the system abstains.
- If the synthesizer's own confidence is below the threshold, it abstains.
- If the synthesis call fails or returns unparseable output, the system falls back to merging verdict files weighted by verdict confidence, and keeps the abstention rules above.
- An abstention still records the files it would have suggested, so the evaluation can measure whether abstaining was the right call.

## Evaluation

**Inputs.** The committed manifest, the issue text from the Postgres corpus, the Qdrant index, and a model config.

**Case selection.** `--set dev` or `--set full`, then an optional `--limit n` drawn by `random.Random(seed).sample` and sorted by issue number, so a subset is reproducible from the seed. The runner refuses `--set full` unless `--allow-full` is passed, which keeps an accidental 500-case spend out of reach.

**Leakage control in retrieval.** The index contains every corpus issue, including the evaluation cases. For each case, retrieval excludes the case's own issue, any candidate fixed by the case's own PR, and any candidate whose fixing PR number is greater than or equal to the case's PR number. PR numbers are assigned at creation, so this is a proxy for "fixed before this fix existed", not an exact merge-time ordering. It is conservative in the direction that matters: a PR opened after the case's fix cannot have been its precedent.

**Metrics.** Let `P` be the ranked predicted files and `A` the answer key.

| Metric | Definition |
|---|---|
| hit@1, hit@3 | fraction of cases where `P[:k]` contains any file in `A` |
| precision@3, recall@3 | mean over cases of `|P[:3] & A| / |P[:3]|` (0 when `P` is empty) and `|P[:3] & A| / |A|` |
| abstention rate | fraction of cases that abstained |
| accuracy when answering | hit@3 over cases that did not abstain |
| confident-but-wrong rate | fraction of all cases that answered and missed at hit@3 |
| precedent available | fraction of cases where any retrieved candidate's fix touched a file in `A` |
| correct abstention rate | of abstentions, the fraction where no precedent was available |
| latency | median and 95th percentile seconds per case |
| cost | summed tokens per tier and dollars where prices are configured |

hit@k and precision/recall are computed on the would-be suggestion for every case, and separately for answered cases only, so abstention cannot hide or inflate them.

**Output.** `eval/results/{set}-n{count}-seed{seed}-{UTC timestamp}/` holding `summary.json` (metrics, model names per tier, case count, seed, spend, token counts, manifest `generated_at`) and `cases.jsonl` (one line per case with predictions, citations, confidence, abstention, latency and tokens). Every reported number comes from these files.

**Cost guardrail.** `--max-cost` (dollars) stops the run once accumulated spend reaches it and records `stopped_early: true`. `--estimate` prints an upper-bound estimate from case count, the prompt character budgets, a 4-characters-per-token heuristic and the configured prices, without calling any model.

## CLI

```
miresolver resolve "title" --body "..." --base-commit SHA [--repo owner/name]
miresolver resolve --issue 12345 --base-commit SHA          (text read from Postgres)
miresolver evaluate --set dev [--limit 20] [--seed 1] [--max-cost 3] [--estimate]
```

## Testing

- Each adapter against `httpx.MockTransport`: request shape, auth header, usage parsing, errors raised as `ChatError`.
- JSON extraction tolerant of code fences and surrounding prose.
- Screen, investigate and synthesize nodes with a scripted fake `ChatClient`.
- The graph end to end with fakes: concurrency (investigators overlap in time), a timed-out investigator returning a partial result within the deadline, the empty-candidate path, abstention thresholds, citation filtering.
- Metrics against hand-computed values, and a golden file for the summary shape.
- The evaluation runner end to end with fakes, including seeded subset selection, the spend cap, and `--set full` refusal.
- CLI wiring with the runner and resolver replaced by doubles.
