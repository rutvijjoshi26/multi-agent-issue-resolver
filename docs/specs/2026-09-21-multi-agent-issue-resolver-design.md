# Multi-Agent Issue Resolver: Design

**Date:** 2026-09-21
**Status:** Approved design, pending implementation plan

## Problem

When a new issue lands on a busy repository, someone has to work out what is going on. Often the answer already exists: the same bug, or one close enough to matter, was diagnosed and fixed months ago by someone who has since moved on. That knowledge lives in closed issue threads and merged pull requests, and nobody reads those.

The current source code does not carry it. Code shows the end state with every dead end erased. It cannot tell you that a bug which looks like a parsing failure turned out three times before to be a file watcher firing early, and that each fix went into the watcher.

This system retrieves that institutional memory and turns it into a concrete suggestion for a new issue.

## Goal

Two goals, weighted equally.

**Product goal.** Given a new issue, return a ranked suggestion naming the files likely involved and the approach that worked for similar past issues, with citations to the specific issues and PRs, or an explicit "no confident match" when nothing fits.

**Evidence goal.** The project exists to produce defensible evidence for capabilities that job postings repeatedly ask for and that the author's resume cannot currently claim:

| Capability asked for | Evidence this project produces |
|---|---|
| LLM evaluation harness | Config-matrix harness over 500 real issues with a free answer key |
| RAG benchmarking (latency, relevance, accuracy, cost) | All four measured per configuration, with baselines |
| A/B and experimental comparison | Three head-to-head experiments, results reported whichever way they fall |
| SLOs, monitoring, alerting | Defined SLOs, Prometheus metrics, alerts on rolling breach |
| Responsible AI, audit logging | Every answer logged with citations and confidence; abstention measured |
| Distributed and parallel systems | Go coordinator with fan-out, timeouts, cancellation, partial results |

Any design decision that does not serve one of these two goals is out of scope.

## Non-goals

- Not a code generation tool. It suggests an approach and cites precedent; it does not open pull requests.
- Not a general chat interface over the repository.
- Not multi-tenant or access-controlled. Public repositories only.
- No sharded vector store in Phase 1. At single-repo scale one node is sufficient, and building scatter-gather anyway would be engineering theater. Sharding enters only in Phase 2, justified by a number.

## The central experiment

The project's premise is falsifiable and should be tested rather than assumed.

A capable coding agent with code search can already locate relevant code. If past-issue retrieval adds nothing on top of that, the product has no reason to exist. So the headline experiment is:

> **Does institutional memory beat raw code search, and on which kinds of issues?**

- **Baseline arm:** an agent with code search over the repository, no issue history.
- **Treatment arm:** the same agent plus retrieved past issues and their fixes.

Both run over the same evaluation set. Results are broken out by issue type (simple localized bug, regression, cross-module) because the expectation is that the treatment wins on regressions and ties on simple bugs. If the treatment does not win anywhere, that is the finding and it gets reported.

Two secondary experiments, both free once the harness exists:

1. **Retrieval strategy:** vector-only versus keyword-only versus hybrid.
2. **Fan-out value:** N parallel investigation agents versus one agent given all candidates in a single prompt. Measured on accuracy, wall-clock time, and cost, because the fan-out costs roughly N times as much and has to justify that.

## Architecture

```
                    new issue text
                          |
                 [Go coordinator]
                          |
              hybrid search (Qdrant)
                          |
                 ~10 candidate issues
                          |
          [cheap screening pass, one batched call]
                          |
                  top 3 candidates
                          |
        +-----------------+-----------------+
        v                 v                 v
   investigator      investigator      investigator   (parallel, Go goroutines,
   agent 1           agent 2           agent 3         per-agent timeout + cancel)
        |                 |                 |
        +-----------------+-----------------+
                          |
                    [synthesizer]
                          |
         suggestion + citations + confidence
                  (or abstention)
```

**Retrieval cascade.** Hybrid search returns roughly ten candidates. A cheap model screens all of them in one batched call and scores relevance. Only the top three get a deep investigation agent. This caps cost regardless of how many candidates the search returns, which is the answer to "what if there are ten candidates."

**Investigator agent.** Each one takes a single past issue plus its fix, examines the current code in the areas that fix touched, and returns a structured verdict: does this precedent apply, which files are implicated now, what approach it suggests, and a confidence score with reasoning.

**Synthesizer.** Weighs the verdicts, merges overlapping file suggestions, produces the final ranked answer with citations. Abstains when no verdict clears the confidence threshold.

**Go coordinator.** Owns fan-out, per-agent timeouts, cancellation of in-flight work when the deadline passes, and returning partial results when an agent fails or straggles. This is where the concurrency engineering lives.

**Python workers.** Embedding, hybrid search, code search, and model calls. Communicates with the coordinator over gRPC.

## Model abstraction (bring your own model)

A hard requirement, not a convenience: users must be able to run this against any model they choose, including a private internal one, without touching code. Many organisations cannot send source code or issue text to a public API at all, so this is often the difference between the system being usable and not.

**Adapters.**

| Adapter | Covers |
|---|---|
| `openai-compatible` | vLLM, Ollama, LM Studio, Together, Groq, Azure OpenAI, and most internal company gateways. User supplies base URL, API key, and model name. |
| `anthropic` | Native, since the API differs enough to be worth a direct implementation |
| `gemini` | Native, same reason |

**Configuration.** Provider and model are set per tier in config. Changing either is a one-line edit with no code change and no redeploy.

```yaml
tiers:
  screen:      { provider: openai-compatible, base_url: http://internal-llm:8000/v1, model: <model-id> }
  investigate: { provider: anthropic, model: <model-id> }
  synthesize:  { provider: openai, model: <model-id> }
```

**Consequence for evaluation.** Because a tier's model is config, any model becomes an arm in the comparison matrix for free. Questions like "is a cheap open model good enough for screening" need a config file, not new code.

## Data stores

| Store | Holds | Why this one |
|---|---|---|
| Qdrant | Issue embeddings and payloads | Native hybrid search, self-hosts in one container, in demand in postings, and not a repeat of Pinecone |
| Postgres | Issues, PR links, agent verdicts, eval runs and results | Relational data with real queries against it |
| Redis | Embedding cache, repeated investigation cache | Makes re-running the eval affordable; produces a measurable cost reduction |

**What gets embedded:** issue title, body, and early comments, which is symptom text. The resolution, PR diff, and changed file list ride along as payload rather than being embedded, so that new symptoms match old symptoms rather than old solutions.

## Evaluation harness

The harness is the centerpiece, not an afterthought.

**Evaluation set.** 100 issues for iteration, 500 for the published run, drawn from the same pool under the rules below.

**Answer key.** Closed issues with a linked merged PR. The PR's changed files are the correct answer. Nobody wrote these to make the system look good.

**Leakage control, the critical correctness requirement.** For each evaluation issue, the repository must be indexed at the commit immediately before the fix landed. Indexing the current repository would mean the fix is already present in the code, inflating every number and making the result meaningless. Snapshots are cached and reused across configurations, since this is the most expensive part of the harness.

**Inclusion rules, stated in the README rather than applied quietly:**
- Issue has a bug label
- Issue links to exactly one merged PR
- PR touches between one and five source files (excludes mass reformatting)
- PR is not a revert

**Metrics, all reported in plain language:**
- How often the correct file appeared in the top 3 (and top 1)
- How often the system correctly abstained when no good precedent existed
- How often it confidently suggested something wrong, which is the failure that matters most
- Seconds per answer, at median and 95th percentile
- Cost per answer in dollars

**Judged sample.** File-matching is automatic. Whether the suggested *approach* was actually right is graded on a sample by hand, with the sample size stated. No LLM grading LLM output without human validation of the grader.

## Operational layer

Defined SLOs, not decorative dashboards:

- 95% of answers return in under 3 seconds
- Correct-file-in-top-3 rate stays above the threshold established by the eval
- Confident-but-wrong rate stays below its threshold

Prometheus metrics for latency, cost, abstention rate, and agent failures. Alerts fire when a rolling window breaches an SLO. Every answer is written to an audit log with its citations, confidence, and which agents contributed, so any suggestion can be reconstructed after the fact.

## Phases

**Phase 1.** Ingestion of one repository, hybrid search in Qdrant, the retrieval cascade, parallel investigators under the Go coordinator, the synthesizer with abstention, the evaluation harness with leakage-controlled snapshots, and the operational layer. Single machine, Docker Compose. This phase alone is a complete, defensible project.

**Phase 2.** Multi-repository ingestion at a scale that genuinely breaks a single node, sharded vectors with scatter-gather in the coordinator, and duplicate detection using GitHub's own duplicate labels as a second free answer key.

**Phase 3, optional.** A GitHub App that comments on new issues, turning the eval into a live system with real feedback.

## Testing

- Unit tests on chunking, hybrid search merging, verdict aggregation, and the abstention threshold
- Golden-file test on the eval pipeline so scoring changes are visible in a diff
- Fault injection killing an investigator mid-flight, asserting partial results return within the deadline
- A small fixture repository with hand-built issues and fixes, so the harness can be tested without hitting the network

## Risks

| Risk | Mitigation |
|---|---|
| Snapshot indexing dominates runtime | Cache snapshots keyed by commit; build the cache before the config matrix runs |
| LLM cost during evaluation | Cheap model for screening and investigation, strong model only for synthesis; Redis cache; cap the matrix |
| Answer key is noisier than it looks | Inclusion rules above, stated publicly |
| The central experiment fails | That is a legitimate result and gets reported. A project that honestly reports a negative finding is more credible than one that only reports wins |
| Scope growth | Phase 1 is independently shippable; nothing in Phase 2 is required for the project to stand |

## Resolved decisions

**Phase 1 repository: microsoft/vscode.** Roughly 180k closed issues, disciplined issue-to-PR linking, consistent bug labelling, and a checkout around 1GB. Link discipline matters more than raw volume here, because it directly determines how clean the answer key is.

**Models: user's choice, tiers are config.** See the model abstraction section. The project ships defaults but hard-codes nothing.

**Evaluation set: tiered.** A 100-issue subset for the development loop, and a single 500-issue run for the numbers that get published. Snapshots are cached and shared between the two, so the small set is a genuine subset of the large one and the expensive indexing happens once.
