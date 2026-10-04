"""Exercises AnthropicClient against a FAKE SDK object shaped like the Messages API response.
This checks our parsing/retry/arg handling only. It does NOT prove the real SDK or API behave this way:
run the smoke test in docs/RUNNING_EXPERIMENTS.md."""
import failurelab.llm as llm
from failurelab.llm import AnthropicClient, ModelError

import pytest


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Resp:
    def __init__(self, content, stop="tool_use"):
        self.content, self.stop_reason = content, stop
        self.usage = _Block(input_tokens=11, output_tokens=7)


class _ApiError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _FakeSDK:
    def __init__(self, outcomes):
        self.outcomes, self.kwargs = list(outcomes), []
        self.messages = self

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _resp():
    return _Resp([_Block(type="text", text="thinking"), _Block(type="tool_use", id="tu_1", name="run_python", input={"code": "1"})])


def test_response_is_parsed_into_blocks_calls_and_usage():
    sdk = _FakeSDK([_resp()])
    turn = AnthropicClient("some-model", client=sdk).complete("sys", [{"role": "user", "content": []}], [{"name": "t"}])
    assert turn.text == "thinking" and turn.stop_reason == "tool_use"
    assert [(c.id, c.name, c.input) for c in turn.tool_calls] == [("tu_1", "run_python", {"code": "1"})]
    assert turn.blocks[1] == {"type": "tool_use", "id": "tu_1", "name": "run_python", "input": {"code": "1"}}
    assert turn.usage == {"input_tokens": 11, "output_tokens": 7}


def test_request_arguments_and_optional_temperature():
    sdk = _FakeSDK([_resp(), _resp()])
    AnthropicClient("m", client=sdk, max_tokens=123).complete("S", [], [])
    AnthropicClient("m", client=sdk, temperature=0.5).complete("S", [], [])
    assert sdk.kwargs[0]["model"] == "m" and sdk.kwargs[0]["max_tokens"] == 123 and sdk.kwargs[0]["system"] == "S"
    assert "temperature" not in sdk.kwargs[0] and sdk.kwargs[1]["temperature"] == 0.5


def test_retryable_errors_are_retried_then_succeed():
    sleeps, real = [], llm.time.sleep
    llm.time.sleep = lambda s: sleeps.append(s)
    try:
        sdk = _FakeSDK([_ApiError(429), _ApiError(529), _resp()])
        turn = AnthropicClient("m", client=sdk).complete("S", [], [])
    finally:
        llm.time.sleep = real
    assert len(sdk.kwargs) == 3 and len(sleeps) == 2 and turn.tool_calls


def test_non_retryable_and_exhausted_errors_abort_the_run_not_the_task():
    real = llm.time.sleep
    llm.time.sleep = lambda s: None
    try:
        with pytest.raises(ModelError):
            AnthropicClient("m", client=_FakeSDK([_ApiError(400)])).complete("S", [], [])
        with pytest.raises(ModelError):
            AnthropicClient("m", client=_FakeSDK([_ApiError(503)] * 9), max_retries=2).complete("S", [], [])
    finally:
        llm.time.sleep = real
