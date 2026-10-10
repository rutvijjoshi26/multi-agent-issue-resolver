"""Test doubles for the agent layer. No network, no real models."""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence

from miresolver.models.chat import ChatError, ChatResult, Usage
from miresolver.retrieval.index import SearchHit


class ScriptedChat:
    """Replies from a queue, or from a function of the prompt.

    Each reply is a string, a dict (sent as JSON), or an exception to raise.
    Every call is recorded so tests can assert on the prompts sent.
    """

    def __init__(self, replies=None, *, respond: Callable | None = None,
                 usage: Usage = Usage(10, 5), delay: float = 0.0) -> None:
        self.replies = list(replies or [])
        self.respond = respond
        self.usage = usage
        self.delay = delay
        self.calls: list[dict] = []

    async def complete(self, system: str, messages: Sequence[Mapping[str, str]],
                       *, max_tokens: int = 1024, json_mode: bool = False) -> ChatResult:
        self.calls.append({"system": system, "messages": list(messages),
                           "json_mode": json_mode})
        prompt = messages[-1]["content"]
        reply = self.respond(prompt) if self.respond else self.replies.pop(0)
        if self.delay:
            await asyncio.sleep(self.delay)
        if callable(reply) and not isinstance(reply, (str, dict)):
            reply = await reply()
        if isinstance(reply, BaseException):
            raise reply
        text = json.dumps(reply) if isinstance(reply, dict) else reply
        return ChatResult(text=text, usage=self.usage)

    async def aclose(self) -> None:
        return None


class FakeReader:
    def __init__(self, files: Mapping[tuple[str, str], str] | None = None,
                 errors: Sequence[str] = ()) -> None:
        self.files = dict(files or {})
        self.errors = set(errors)
        self.reads: list[tuple[str, str]] = []

    async def read(self, commit: str, path: str) -> str | None:
        self.reads.append((commit, path))
        if path in self.errors:
            from miresolver.code.reader import CodeReadError
            raise CodeReadError(f"cannot read {path}")
        return self.files.get((commit, path))

    async def aclose(self) -> None:
        return None


def hit(number: int, *, pr: int | None = None, files=("src/a.ts",),
        title: str | None = None, body: str = "symptom text") -> SearchHit:
    return SearchHit(
        issue_number=number,
        score=1.0 / number,
        payload={
            "issue_number": number,
            "title": title or f"issue {number}",
            "body": body,
            "pr_number": pr if pr is not None else number + 1000,
            "changed_files": tuple(files),
            "base_commit": "f" * 40,
        },
    )


__all__ = ["ScriptedChat", "FakeReader", "hit", "ChatError", "Usage"]
