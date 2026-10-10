from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from miresolver.models.chat import Usage
from miresolver.retrieval.index import SearchHit


@dataclass(frozen=True)
class NewIssue:
    """The bug report to resolve, and the commit whose code it is about."""

    title: str
    body: str
    base_commit: str
    number: int | None = None


@dataclass(frozen=True)
class Candidate:
    """A solved past issue that retrieval proposed as a precedent."""

    issue_number: int
    title: str
    body: str
    pr_number: int | None
    changed_files: tuple[str, ...]
    retrieval_rank: int
    screen_score: float | None = None

    @classmethod
    def from_hit(cls, hit: SearchHit, rank: int) -> "Candidate":
        payload = hit.payload
        pr = payload.get("pr_number")
        return cls(
            issue_number=hit.issue_number,
            title=str(payload.get("title") or ""),
            body=str(payload.get("body") or ""),
            pr_number=int(pr) if isinstance(pr, int) else None,
            changed_files=tuple(str(f) for f in payload.get("changed_files") or ()),
            retrieval_rank=rank,
        )


@dataclass(frozen=True)
class Verdict:
    """One investigator's judgement of one precedent."""

    issue_number: int
    pr_number: int | None
    applies: bool
    confidence: float
    files: tuple[str, ...]
    approach: str
    reasoning: str
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    @classmethod
    def failure(cls, candidate: Candidate, error: str) -> "Verdict":
        return cls(
            issue_number=candidate.issue_number,
            pr_number=candidate.pr_number,
            applies=False,
            confidence=0.0,
            files=(),
            approach="",
            reasoning="",
            error=error,
        )


@dataclass(frozen=True)
class Citation:
    issue_number: int
    pr_number: int | None


@dataclass(frozen=True)
class Resolution:
    """The system's answer for one issue.

    When abstained is true, files and approach still hold what the system
    would have suggested, so an evaluation can judge whether abstaining was
    the right call. They must not be shown to a user as a recommendation.
    """

    approach: str
    files: tuple[str, ...]
    citations: tuple[Citation, ...]
    confidence: float
    abstained: bool
    abstain_reason: str | None
    candidates: tuple[Candidate, ...] = ()
    screened: tuple[Candidate, ...] = ()
    verdicts: tuple[Verdict, ...] = ()
    usage: Mapping[str, Usage] = field(
        default_factory=lambda: MappingProxyType({})
    )
    #: True when synthesis failed and files were merged from the verdicts.
    synthesis_fallback: bool = False
    #: Model calls whose token usage is unknown (an investigator that timed
    #: out or crashed may still have been billed), so spend is a floor.
    unmetered_calls: int = 0


@dataclass(frozen=True)
class ResolverOptions:
    retrieve_k: int = 10
    investigate_n: int = 3
    screen_floor: float = 0.2
    #: Minimum confidence of an applicable verdict for the system to answer.
    abstain_threshold: float = 0.5
    #: Minimum synthesizer confidence to answer; None means abstain_threshold.
    synthesis_threshold: float | None = None
    #: Never abstain. Used to report coverage-style metrics alongside the
    #: abstaining mode; the would-be abstention reason is still recorded.
    always_answer: bool = False
    #: "hybrid" (dense and BM25 fused by rank), "dense" or "keyword".
    retrieval: str = "hybrid"
    investigator_timeout_s: float = 90.0
    file_char_budget: int = 6000
    max_files_per_candidate: int = 5
    candidate_body_chars: int = 600
