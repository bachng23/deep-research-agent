import pytest

from paper_research_agent.llm import invoke_with_retry


class _FlakyModel:
    """A structured model whose .invoke fails the first `fail_times` calls."""

    def __init__(self, fail_times: int, result="ok"):
        self.calls = 0
        self.fail_times = fail_times
        self.result = result

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError("malformed structured output")
        return self.result


def test_retry_succeeds_after_transient_failures():
    model = _FlakyModel(fail_times=2)

    result = invoke_with_retry(model, [], retries=2)

    assert result == "ok"
    assert model.calls == 3  # 2 failures + 1 success


def test_retry_returns_immediately_on_first_success():
    model = _FlakyModel(fail_times=0)

    result = invoke_with_retry(model, [], retries=2)

    assert result == "ok"
    assert model.calls == 1  # no wasted retries


def test_retry_raises_after_exhausting_attempts():
    model = _FlakyModel(fail_times=99)

    with pytest.raises(ValueError, match="malformed structured output"):
        invoke_with_retry(model, [], retries=2)

    assert model.calls == 3  # initial attempt + 2 retries


class _EmptyThenFullModel:
    """Returns an invalid (empty) result the first `empty_times` calls."""

    def __init__(self, empty_times: int):
        self.calls = 0
        self.empty_times = empty_times

    def invoke(self, messages):
        self.calls += 1
        return [] if self.calls <= self.empty_times else ["gap"]


def test_retry_on_invalid_result_then_succeeds():
    model = _EmptyThenFullModel(empty_times=1)

    result = invoke_with_retry(model, [], retries=2, is_valid=lambda r: bool(r))

    assert result == ["gap"]
    assert model.calls == 2  # 1 invalid (empty) + 1 valid


def test_retry_raises_when_result_stays_invalid():
    model = _EmptyThenFullModel(empty_times=99)

    with pytest.raises(ValueError, match="validation"):
        invoke_with_retry(model, [], retries=2, is_valid=lambda r: bool(r))

    assert model.calls == 3  # initial + 2 retries, all invalid
