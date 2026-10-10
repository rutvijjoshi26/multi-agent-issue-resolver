"""The resolution workflow as a LangGraph state graph.

retrieve -> screen -> investigate (one Send per screened candidate, run
concurrently) -> synthesize. Each investigator runs under its own deadline;
one that fails or straggles contributes a failed verdict instead of
failing the run, so the synthesizer always works with partial results.
"""

import asyncio
import operator
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from miresolver.agents.investigate import investigate
from miresolver.agents.screen import screen
from miresolver.agents.synthesize import SynthesisResult, synthesize
from miresolver.agents.types import (
    Candidate,
    NewIssue,
    Resolution,
    ResolverOptions,
    Verdict,
)
from miresolver.code.reader import CodeReader
from miresolver.models.chat import ChatClient, Usage
from miresolver.retrieval.documents import MAX_EMBED_CHARS
from miresolver.retrieval.index import SearchHit

Retriever = Callable[[str, int], Awaitable[list[SearchHit]]]
Exclude = Callable[[Candidate], bool]

#: Hits requested per wanted candidate, so that unsolved issues and
#: excluded ones are backfilled up to retrieve_k.
OVERFETCH = 4
#: When too few hits survive the filters, the request grows by this factor
#: and is retried, up to MAX_RETRIEVE hits. Most corpus issues have no merged
#: fix and the leakage filter removes more, so for an older case only a few
#: percent of hits may be usable precedents.
WIDEN_FACTOR = 4
MAX_RETRIEVE = 4096


@dataclass(frozen=True)
class ResolverDeps:
    retriever: Retriever
    screen_chat: ChatClient
    investigate_chat: ChatClient
    synthesize_chat: ChatClient
    reader: CodeReader
    options: ResolverOptions = field(default_factory=ResolverOptions)


class ResolveState(TypedDict, total=False):
    issue: NewIssue
    exclude: Exclude | None
    candidates: list[Candidate]
    screened: list[Candidate]
    candidate: Candidate
    verdicts: Annotated[list[Verdict], operator.add]
    usage: Annotated[list[tuple[str, Usage]], operator.add]
    unmetered: Annotated[int, operator.add]
    synthesis: SynthesisResult | None
    abstain_reason: str | None


def query_text(issue: NewIssue) -> str:
    """Same symptom text the index embeds: title and body."""
    return f"{issue.title}\n\n{issue.body}".strip()[:MAX_EMBED_CHARS]


async def retrieve_candidates(
    retriever: Retriever, issue: NewIssue, *, k: int, exclude: Exclude | None
) -> list[Candidate]:
    """Up to k usable precedents, in retrieval order.

    Shared by the workflow and the retrieval-only check, so both see
    exactly the same candidates under the same filters.
    """
    query = query_text(issue)
    limit = k * OVERFETCH
    while True:
        hits = await retriever(query, limit)
        candidates = [Candidate.from_hit(h, rank) for rank, h in enumerate(hits)]
        # The index keeps issues with no merged fix as symptom matches,
        # but they carry no resolution, so they are not precedents.
        candidates = [
            c for c in candidates if c.pr_number is not None and c.changed_files
        ]
        if exclude:
            candidates = [c for c in candidates if not exclude(c)]
        # Stop once k survive, the index has nothing more to give, or the
        # cap is reached.
        if len(candidates) >= k or len(hits) < limit or limit >= MAX_RETRIEVE:
            break
        limit = min(limit * WIDEN_FACTOR, MAX_RETRIEVE)
    return [replace(c, retrieval_rank=rank) for rank, c in enumerate(candidates[:k])]


def build_graph(deps: ResolverDeps) -> Any:
    options = deps.options

    async def retrieve_node(state: ResolveState) -> dict:
        candidates = await retrieve_candidates(
            deps.retriever, state["issue"], k=options.retrieve_k,
            exclude=state.get("exclude"),
        )
        return {"candidates": candidates, "verdicts": [], "usage": []}

    async def screen_node(state: ResolveState) -> dict:
        kept, usage = await screen(
            state["issue"],
            state["candidates"],
            deps.screen_chat,
            top_n=options.investigate_n,
            floor=options.screen_floor,
            body_chars=options.candidate_body_chars,
        )
        return {"screened": kept, "usage": [("screen", usage)]}

    def fan_out(state: ResolveState) -> list[Send] | str:
        if not state["screened"]:
            return "synthesize"
        return [
            Send("investigate", {"issue": state["issue"], "candidate": c})
            for c in state["screened"]
        ]

    async def investigate_node(state: ResolveState) -> dict:
        candidate = state["candidate"]
        try:
            verdict, usage = await asyncio.wait_for(
                investigate(
                    state["issue"], candidate, deps.investigate_chat,
                    deps.reader, options,
                ),
                timeout=options.investigator_timeout_s,
            )
        except TimeoutError:
            verdict, usage = (
                Verdict.failure(
                    candidate,
                    f"timed out after {options.investigator_timeout_s:g}s",
                ),
                Usage(),
            )
        except Exception as exc:  # one investigator must not sink the run
            verdict, usage = (
                Verdict.failure(candidate, f"{type(exc).__name__}: {exc}"),
                Usage(),
            )
        return {
            "verdicts": [verdict],
            "usage": [("investigate", usage)],
            "unmetered": 1 if verdict.failed and usage == Usage() else 0,
        }

    async def synthesize_node(state: ResolveState) -> dict:
        if not state["candidates"]:
            return {"synthesis": None, "abstain_reason": "no_candidates"}
        if not state["screened"]:
            return {"synthesis": None, "abstain_reason": "no_candidates_after_screening"}
        order = {c.issue_number: i for i, c in enumerate(state["screened"])}
        verdicts = sorted(state["verdicts"], key=lambda v: order[v.issue_number])
        result = await synthesize(
            state["issue"], verdicts, deps.synthesize_chat,
            threshold=options.abstain_threshold,
            synthesis_threshold=options.synthesis_threshold,
            always_answer=options.always_answer,
            min_agreement=options.min_agreement,
        )
        return {
            "synthesis": result,
            "verdicts": [],
            "usage": [("synthesize", result.usage)],
            "abstain_reason": result.reason,
        }

    graph = StateGraph(ResolveState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("screen", screen_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        lambda s: "screen" if s["candidates"] else "synthesize",
        ["screen", "synthesize"],
    )
    graph.add_conditional_edges("screen", fan_out, ["investigate", "synthesize"])
    graph.add_edge("investigate", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def _sum_usage(entries: list[tuple[str, Usage]]) -> dict[str, Usage]:
    totals: dict[str, Usage] = {}
    for tier, usage in entries:
        totals[tier] = totals.get(tier, Usage()) + usage
    return totals


async def resolve(
    issue: NewIssue,
    deps: ResolverDeps,
    *,
    exclude: Exclude | None = None,
    graph: Any | None = None,
) -> Resolution:
    """Run the workflow for one issue. Pass a prebuilt graph to reuse it."""
    app = graph if graph is not None else build_graph(deps)
    state = await app.ainvoke(
        {"issue": issue, "exclude": exclude, "screened": [], "unmetered": 0}
    )
    order = {c.issue_number: i for i, c in enumerate(state.get("screened") or [])}
    verdicts = tuple(
        sorted(state.get("verdicts") or [], key=lambda v: order.get(v.issue_number, 0))
    )
    common = dict(
        candidates=tuple(state.get("candidates") or ()),
        screened=tuple(state.get("screened") or ()),
        verdicts=verdicts,
        usage=MappingProxyType(_sum_usage(state.get("usage") or [])),
        unmetered_calls=state.get("unmetered") or 0,
    )
    synthesis: SynthesisResult | None = state.get("synthesis")
    if synthesis is None:
        return Resolution(
            approach="", files=(), citations=(), confidence=0.0,
            abstained=not deps.options.always_answer,
            abstain_reason=state.get("abstain_reason"), **common,
        )
    return Resolution(
        approach=synthesis.approach,
        files=synthesis.files,
        citations=synthesis.citations,
        confidence=synthesis.confidence,
        abstained=synthesis.abstained,
        abstain_reason=synthesis.reason,
        synthesis_fallback=synthesis.fallback,
        agreement=synthesis.agreement,
        **common,
    )
