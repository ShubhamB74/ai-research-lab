"""Model-client layer: a tiny neutral interface plus an Anthropic adapter and a scripted stand-in.

Messages use Anthropic-style content blocks:
  assistant: [{"type": "text", ...}, {"type": "tool_use", "id", "name", "input"}]
  user     : [{"type": "text", ...}] or [{"type": "tool_result", "tool_use_id", "content", "is_error"?}]

IMPORTANT: `AnthropicClient` has NOT been exercised in the environment where this harness was built
(no network, no API key). Everything else is tested with `ScriptedClient`. Run the smoke test in
docs/RUNNING_EXPERIMENTS.md before a full experiment.
"""
from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


class RunAborted(BaseException):
    """Stops a whole run (budget hit, API outage). Deliberately a BaseException so the runner's
    `except Exception` does NOT score it as an agent failure. Re-running the same command resumes."""


class BudgetExceeded(RunAborted):
    pass


class ModelError(RunAborted):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ModelTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class ModelClient(ABC):
    name: str

    @abstractmethod
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn: ...


class BudgetedClient(ModelClient):
    """Hard cap on total model calls across a whole experiment."""

    def __init__(self, inner: ModelClient, max_calls: int):
        self.inner, self.max_calls, self.calls = inner, max_calls, 0
        self.name = inner.name

    def complete(self, system, messages, tools) -> ModelTurn:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"model-call budget of {self.max_calls} reached; re-run to resume")
        self.calls += 1
        return self.inner.complete(system, messages, tools)


class AnthropicClient(ModelClient):
    RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}

    def __init__(self, model: str, *, temperature: float | None = None, max_tokens: int = 4096,
                 max_retries: int = 4, client: Any = None):
        self.model, self.temperature, self.max_tokens, self.max_retries = model, temperature, max_tokens, max_retries
        self.name = f"anthropic:{model}"
        if client is None:
            import anthropic  # optional dependency: pip install -e ".[llm]"
            client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
        self.client = client

    def complete(self, system, messages, tools) -> ModelTurn:
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      messages=messages, tools=tools)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.messages.create(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                status = getattr(exc, "status_code", None)
                retryable = status in self.RETRYABLE or type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
                if not retryable or attempt == self.max_retries:
                    raise ModelError(f"model call failed: {type(exc).__name__}: {exc}") from exc
                time.sleep(min(60, 2 ** (attempt + 1)))
        blocks, calls, text = [], [], []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text}); text.append(b.text)
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
                calls.append(ToolCall(b.id, b.name, dict(b.input)))
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", 0),
                 "output_tokens": getattr(resp.usage, "output_tokens", 0)}
        return ModelTurn("".join(text), calls, blocks, resp.stop_reason or "", usage)


# ---------------------------------------------------------------------- scripted stand-in
_ids = itertools.count(1)


def tool_turn(*calls: tuple[str, dict], text: str = "") -> ModelTurn:
    """Build a ModelTurn that makes tool calls, e.g. tool_turn(("run_python", {"code": "..."}))."""
    tcs = [ToolCall(f"toolu_scripted_{next(_ids)}", name, inp) for name, inp in calls]
    blocks = ([{"type": "text", "text": text}] if text else []) + [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in tcs]
    return ModelTurn(text, tcs, blocks, "tool_use")


def text_turn(text: str) -> ModelTurn:
    return ModelTurn(text, [], [{"type": "text", "text": text}], "end_turn")


class ScriptedClient(ModelClient):
    """A deterministic fake 'model' driven by policy(messages, tools) -> ModelTurn.
    Used to test the harness end to end without any API. It tests plumbing, NOT LLM behaviour."""

    def __init__(self, name: str, policy: Callable[[list[dict], list[dict]], ModelTurn]):
        self.name, self.policy, self.calls = name, policy, 0

    def complete(self, system, messages, tools) -> ModelTurn:
        self.calls += 1
        return self.policy(messages, tools)
