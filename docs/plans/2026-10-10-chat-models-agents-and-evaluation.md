# Chat Models, Agents and Evaluation Implementation Plan

**Goal:** Given a new bug report, recommend an approach and the files likely involved, citing the solved issues and PRs it relied on or abstaining, and measure that against the committed answer key.

**Architecture:** A `ChatClient` protocol joins the existing `EmbeddingClient` in the model layer, with OpenAI-compatible, Anthropic and Gemini adapters chosen per tier from config. A LangGraph `StateGraph` runs retrieve, screen, a parallel `Send` fan-out of investigators reading code at the pinned base commit, and a synthesizer whose abstention is enforced by deterministic rules. An evaluation runner drives the graph over a seeded slice of the manifest and writes `summary.json` and `cases.jsonl` under `eval/results/`.

**Tech Stack:** Python 3.12+, LangGraph, httpx, qdrant-client, psycopg, PyYAML, pytest with pytest-asyncio.

**Spec:** `docs/specs/2026-10-10-agents-and-evaluation-design.md` (parent: `docs/specs/2026-09-21-multi-agent-issue-resolver-design.md`)

## Global Constraints

- No provider name or model id literal in `src/` outside the adapters' default base URLs. Tiers come from config.
- No network in unit tests: adapters use `httpx.MockTransport`, agents use a scripted fake `ChatClient`, the code reader uses a fake or a temporary git repository.
- Code shown to an investigator is read at the new issue's `base_commit`, never the current tree.
- Abstention is decided by deterministic rules over model outputs, never by the model alone.
- No invented costs: a tier without configured prices reports `null` dollars.
- Tests run with `venv/bin/python -m pytest -q`. Baseline before this plan: 143 passed, 12 skipped.
- Nothing is committed; work stays in the working tree.

## File Map

| File | Responsibility |
|---|---|
| `src/miresolver/models/settings.py` (modify) | Optional `input_cost_per_mtok`, `output_cost_per_mtok`; `TierSettings.cost(usage)` |
| `src/miresolver/models/chat.py` (create) | `Usage`, `ChatResult`, `ChatError`, `ChatClient`, `OpenAICompatibleChat`, `extract_json` |
| `src/miresolver/models/anthropic.py` (modify) | Add `AnthropicChat` |
| `src/miresolver/models/gemini.py` (modify) | Add `GeminiChat` |
| `src/miresolver/models/factory.py` (modify) | `build_chat(settings, tier, env, *, transport=None)` |
| `src/miresolver/code/reader.py` (create) | `CodeReader`, `GitHubRawCodeReader`, `LocalGitCodeReader` |
| `src/miresolver/agents/types.py` (create) | `NewIssue`, `Candidate`, `Verdict`, `Citation`, `Resolution`, `ResolverOptions` |
| `src/miresolver/agents/prompts.py` (create) | Prompt text builders |
| `src/miresolver/agents/screen.py` (create) | Batched screening with retrieval-order fallback |
| `src/miresolver/agents/investigate.py` (create) | One investigator, code at base commit, structured verdict |
| `src/miresolver/agents/synthesize.py` (create) | Synthesis call, deterministic merge and abstention |
| `src/miresolver/agents/graph.py` (create) | `ResolverDeps`, `build_graph`, `resolve` |
| `src/miresolver/evaluation/metrics.py` (create) | Per-case scoring and aggregate summary |
| `src/miresolver/evaluation/runner.py` (create) | Case selection, leakage exclusion, run loop, spend cap, estimate, output files |
| `src/miresolver/evaluation/corpus.py` (create) | Load issue text and candidate payloads from Postgres |
| `src/miresolver/cli.py` (modify) | `resolve` and `evaluate` subcommands |
| `pyproject.toml`, `config/models.example.yaml`, `README.md` (modify) | Dependency, prices, docs |

---

### Task 1: Tier prices and chat protocol with the OpenAI-compatible adapter

**Interfaces produced:**
- `@dataclass(frozen=True) class Usage: input_tokens: int = 0; output_tokens: int = 0` with `__add__`
- `@dataclass(frozen=True) class ChatResult: text: str; usage: Usage`
- `class ChatError(Exception)`
- `class ChatClient(Protocol): async def complete(self, system: str, messages: Sequence[Mapping[str, str]], *, max_tokens: int = 1024, json_mode: bool = False) -> ChatResult; async def aclose(self) -> None`
- `class OpenAICompatibleChat(settings, api_key, *, transport=None)`
- `extract_json(text: str) -> dict` raising `ChatError` when no JSON object is found
- `TierSettings.input_cost_per_mtok: float | None`, `output_cost_per_mtok: float | None`, `TierSettings.cost(usage) -> float | None`

- [ ] Write failing tests in `tests/models/test_chat.py`: request goes to `{base_url}/chat/completions` with bearer auth, model, system message first, `max_tokens`; `json_mode` adds `response_format`; usage parsed from `usage.prompt_tokens/completion_tokens`; HTTP error and missing `choices` raise `ChatError`; `extract_json` handles bare JSON, fenced JSON, prose around JSON, and raises on none. In `tests/models/test_settings.py`: prices parse, `cost` returns `None` without prices and the correct dollar figure with them, a non-numeric price raises `ModelConfigError`.
- [ ] Run them, confirm they fail on import.
- [ ] Implement.
- [ ] Run the full suite, confirm green.

### Task 2: Native Anthropic and Gemini chat adapters and `build_chat`

**Interfaces produced:**
- `AnthropicChat(settings, api_key, *, transport=None)`: `POST /v1/messages`, headers `x-api-key`, `anthropic-version: 2023-06-01`, body `{model, max_tokens, system, messages}`; text is the concatenation of `content[*].text` blocks; usage from `usage.input_tokens/output_tokens`.
- `GeminiChat(settings, api_key, *, transport=None)`: `POST /v1beta/models/{model}:generateContent`, header `x-goog-api-key`, body `{systemInstruction, contents[{role: user|model, parts[{text}]}], generationConfig{maxOutputTokens, responseMimeType?}}`; usage from `usageMetadata.promptTokenCount/candidatesTokenCount`.
- `build_chat(settings: ModelSettings, tier: str, env: Mapping[str, str], *, transport=None) -> ChatClient`; rejects `embed` as a chat tier.

- [ ] Write failing tests `tests/models/test_native_chat.py` and extend `tests/models/test_factory.py`: request shape for each adapter, default and overridden base URL, role mapping (`assistant` to `model` for Gemini), usage, errors; factory returns the right class per provider, raises on unknown provider, missing key, and `embed`.
- [ ] Run, confirm failure. Implement. Run full suite.

### Task 3: Code reader at a pinned commit

**Interfaces produced:**
- `class CodeReader(Protocol): async def read(self, commit: str, path: str) -> str | None; async def aclose(self) -> None`
- `GitHubRawCodeReader(repo: str, cache_dir: Path, *, transport=None, base_url="https://raw.githubusercontent.com")`: 404 gives `None`, other errors raise `CodeReadError`; cached files and cached misses are served without a request.
- `LocalGitCodeReader(repo_path: Path)`: `git show commit:path`, missing path gives `None`.
- Paths containing `..` or starting with `/` and commits that are not hex are rejected with `CodeReadError`.

- [ ] Write failing tests `tests/code/test_reader.py` (MockTransport for raw, a temporary git repository with two commits for local, proving the older commit's content is returned). Run, implement, run suite.

### Task 4: Agent types, prompts, screening and investigation

**Interfaces produced:**
- `NewIssue(title: str, body: str, base_commit: str, number: int | None = None)`
- `Candidate(issue_number: int, title: str, body: str, pr_number: int | None, changed_files: tuple[str, ...], retrieval_rank: int, screen_score: float | None = None)`; `Candidate.from_hit(hit: SearchHit, rank: int)`
- `Verdict(issue_number, pr_number, applies: bool, confidence: float, files: tuple[str, ...], approach: str, reasoning: str, error: str | None = None)` with `failed` property
- `ResolverOptions(retrieve_k=10, investigate_n=3, screen_floor=0.2, abstain_threshold=0.5, investigator_timeout_s=90.0, file_char_budget=6000, max_files_per_candidate=5)`
- `async screen(issue, candidates, chat, *, top_n, floor) -> tuple[list[Candidate], Usage]`
- `async investigate(issue, candidate, chat, reader, options) -> tuple[Verdict, Usage]`

- [ ] Write failing tests `tests/agents/test_screen.py`, `tests/agents/test_investigate.py` with a `ScriptedChat` fake (records prompts, returns queued texts or raises): screen keeps top-n by score above the floor, omitted candidates score 0, scores clamp to 0..1, a failing call falls back to retrieval order with zero usage, empty input makes no call. Investigate reads each changed file at the issue's base commit (asserted via a fake reader), notes missing files in the prompt, truncates to the budget, parses the verdict, clamps confidence, and a malformed reply yields a failed verdict rather than an exception.
- [ ] Run, implement, run suite.

### Task 5: Synthesis and deterministic abstention

**Interfaces produced:**
- `Citation(issue_number: int, pr_number: int | None)`
- `Resolution(approach: str, files: tuple[str, ...], citations: tuple[Citation, ...], confidence: float, abstained: bool, abstain_reason: str | None, candidates: tuple[Candidate, ...], verdicts: tuple[Verdict, ...], usage: Mapping[str, Usage])`
- `merge_verdict_files(verdicts) -> list[str]`: files ranked by summed confidence of applying verdicts, ties by first appearance
- `async synthesize(issue, verdicts, chat, *, threshold) -> tuple[approach, files, citations, confidence, abstained, reason, Usage]` packaged as `SynthesisResult`

- [ ] Write failing tests `tests/agents/test_synthesize.py`: no applying verdict above threshold abstains without calling the model; low synthesizer confidence abstains but keeps would-be files; citations to uninvestigated issues are dropped and PR numbers attached; failed synthesis falls back to merged files with the strongest verdict's confidence; failed verdicts are excluded from the prompt.
- [ ] Run, implement, run suite.

### Task 6: LangGraph workflow

**Interfaces produced:**
- `Retriever = Callable[[str, int], Awaitable[list[SearchHit]]]`
- `ResolverDeps(retriever, screen_chat, investigate_chat, synthesize_chat, reader, options=ResolverOptions())`
- `build_graph(deps) -> CompiledStateGraph`
- `async resolve(issue: NewIssue, deps: ResolverDeps, *, exclude: Callable[[Candidate], bool] | None = None) -> Resolution`

- [ ] Write failing tests `tests/agents/test_graph.py`: end to end with fakes produces cited files; three investigators each sleeping 0.2s finish in well under 0.6s; one investigator that hangs past `investigator_timeout_s` returns a failed verdict while the others' results are used, inside the deadline; zero retrieval hits abstains with reason `no_candidates`; `exclude` removes candidates before screening; usage is summed per tier.
- [ ] Run, implement, run suite.

### Task 7: Metrics

**Interfaces produced:**
- `CaseScore` dataclass with `hit_at_1, hit_at_3, precision_at_3, recall_at_3, abstained, answered_correct, confident_wrong, precedent_available`
- `score_case(predicted: Sequence[str], answer: Sequence[str], *, abstained: bool, candidate_files: Iterable[str]) -> CaseScore`
- `summarize(scores: Sequence[CaseScore], latencies: Sequence[float]) -> dict`

- [ ] Write failing tests `tests/evaluation/test_metrics.py` with hand-computed cases (hit at rank 2, empty prediction, abstention with and without precedent) and a golden file `tests/evaluation/golden_summary.json` for `summarize` over a fixed list. Run, implement, run suite.

### Task 8: Evaluation runner and corpus loader

**Interfaces produced:**
- `select_cases(manifest: Mapping, *, which: str, limit: int | None, seed: int, allow_full: bool) -> list[EvalCase]`
- `leakage_filter(case: EvalCase) -> Callable[[Candidate], bool]`
- `estimate_cost(n_cases, settings, options) -> dict`
- `async run_evaluation(cases, *, issue_text: Callable[[int], tuple[str, str] | None], resolver: Callable[[NewIssue, Callable], Awaitable[Resolution]], settings: ModelSettings, out_dir: Path, run_meta: Mapping, max_cost: float | None) -> Path`
- `load_issue_text(conn, numbers) -> dict[int, tuple[str, str]]` in `corpus.py`

- [ ] Write failing tests `tests/evaluation/test_runner.py`: subset is deterministic per seed and differs across seeds, `full` refused without `allow_full`; leakage filter drops the case itself, its PR, and later PRs; run writes `summary.json` and `cases.jsonl` with model names, counts, tokens and dollars; spend cap stops early and records it; a case whose text is missing is recorded as skipped, not scored; a resolver exception is recorded as an error case. Postgres loader test gated on `TEST_DATABASE_URL`.
- [ ] Run, implement, run suite.

### Task 9: CLI, config example, README

- [ ] Write failing tests `tests/test_cli_agents.py`: `resolve` prints files, citations and confidence or the abstention, using a monkeypatched resolver; `resolve` without `--base-commit` errors; `evaluate --estimate` prints an estimate without building any client; `evaluate --set full` without `--allow-full` exits 1; config errors exit 1 cleanly. Extend `tests/test_readme.py` to require the new flags.
- [ ] Implement CLI wiring, add `langgraph` to `pyproject.toml`, prices and comments to `config/models.example.yaml`, README Status, usage and evaluation sections.
- [ ] Run full suite.

### Task 10: Run the evaluation if credentials allow

- [ ] Check for an LLM key and GitHub token in the environment or `.env` without printing values.
- [ ] If present: bring up services, ingest, index, `evaluate --estimate`, run dev set or a seeded subset under US$3 in the background, poll logs.
- [ ] If absent: record that the evaluation was not run and why, in the README.
