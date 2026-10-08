"""_call_claude_code must name what went wrong instead of handing back "".

`claude -p --output-format json` reports a refusal, an overload or a spent quota
as an rc=0 envelope with `is_error` set; its `result` can also be empty. Both
used to leave the caller parsing an empty string.
"""
import json as _json

import pytest

from backend.analyzer.llm_client import NonRetryableLLMError, _call_claude_code


def _cli_printing(monkeypatch, payload, rc=0, stderr=b""):
    out = payload if isinstance(payload, bytes) else _json.dumps(payload).encode()

    async def fake_run_cli(cmd, stdin, env=None, timeout=None, cwd=None):
        return rc, out, stderr

    monkeypatch.setattr("backend.analyzer.llm_client._run_cli", fake_run_cli)


async def _call():
    return await _call_claude_code("p", "s", "claude-sonnet-5", 3000)


@pytest.mark.asyncio
async def test_agent_features_are_switched_off_and_the_system_message_replaces_the_default(monkeypatch):
    calls = []

    async def fake_run_cli(cmd, stdin, env=None, timeout=None, cwd=None):
        calls.append({"cmd": cmd, "stdin": stdin, "cwd": cwd})
        return 0, _json.dumps({"result": "ok", "subtype": "success"}).encode(), b""

    monkeypatch.setattr("backend.analyzer.llm_client._run_cli", fake_run_cli)
    monkeypatch.setattr("backend.analyzer.llm_client._claude_supports_json_schema", _never)

    await _call_claude_code("the prompt", "the system message", "claude-sonnet-5", 3000)

    cmd = calls[0]["cmd"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert cmd[cmd.index("--setting-sources") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "the system message"
    assert calls[0]["stdin"] == b"the prompt"
    assert "jobnavigator-claude-" in calls[0]["cwd"]


async def _never():
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize("model, reasoning, thinking_off", [
    ("claude-haiku-4-5", False, True),
    ("claude-haiku-4-5", True, False),
    ("claude-sonnet-5-5", False, False),
])
async def test_haiku_thinking_is_switched_off_only_for_low_effort_calls(monkeypatch, model, reasoning, thinking_off):
    environments = []

    async def fake_run_cli(cmd, stdin, env=None, timeout=None, cwd=None):
        environments.append(env)
        return 0, _json.dumps({"result": "ok", "subtype": "success"}).encode(), b""

    monkeypatch.delenv("MAX_THINKING_TOKENS", raising=False)
    monkeypatch.setattr("backend.analyzer.llm_client._run_cli", fake_run_cli)
    monkeypatch.setattr("backend.analyzer.llm_client._claude_supports_json_schema", _never)

    await _call_claude_code("the prompt", "the system message", model, 3000, reasoning=reasoning)

    assert (environments[0].get("MAX_THINKING_TOKENS") == "0") is thinking_off


@pytest.mark.asyncio
async def test_a_normal_envelope_yields_its_result(monkeypatch):
    _cli_printing(monkeypatch, {"result": '{"summary": "x"}', "is_error": False,
                                "subtype": "success"})
    assert (await _call())["text"] == '{"summary": "x"}'


@pytest.mark.asyncio
async def test_plain_stdout_still_passes_through(monkeypatch):
    """Not every build prints an envelope; a bare answer is still an answer."""
    _cli_printing(monkeypatch, b'{"summary": "x"}')
    assert (await _call())["text"] == '{"summary": "x"}'


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"result": "", "is_error": False, "subtype": "success"},
    {"result": "   \n ", "is_error": False},
    b"",
])
async def test_an_empty_reply_raises_instead_of_returning_nothing(monkeypatch, payload):
    _cli_printing(monkeypatch, payload)
    with pytest.raises(RuntimeError) as exc:
        await _call()
    assert "empty reply" in str(exc.value)
    # Retryable on purpose: call_llm's loop should get another go at it.
    assert not isinstance(exc.value, NonRetryableLLMError)


@pytest.mark.asyncio
async def test_an_error_envelope_carries_the_cli_text(monkeypatch):
    _cli_printing(monkeypatch, {"result": "API Error: 500 Internal Server Error",
                                "is_error": True, "subtype": "error_during_execution"})
    with pytest.raises(RuntimeError) as exc:
        await _call()
    assert "500 Internal Server Error" in str(exc.value)


@pytest.mark.asyncio
async def test_an_error_envelope_without_text_still_names_its_subtype(monkeypatch):
    _cli_printing(monkeypatch, {"result": "", "is_error": True,
                                "subtype": "error_max_turns"})
    with pytest.raises(RuntimeError) as exc:
        await _call()
    assert "error_max_turns" in str(exc.value)


@pytest.mark.asyncio
async def test_a_spent_quota_is_not_retried(monkeypatch):
    _cli_printing(monkeypatch, {"result": "Claude usage limit reached — resets at 5pm",
                                "is_error": True, "subtype": "error"})
    with pytest.raises(NonRetryableLLMError) as exc:
        await _call()
    assert "usage limit" in str(exc.value)


@pytest.mark.asyncio
async def test_a_failed_subprocess_still_reports_stderr(monkeypatch):
    _cli_printing(monkeypatch, b"", rc=1, stderr=b"not logged in")
    with pytest.raises(RuntimeError) as exc:
        await _call()
    assert "not logged in" in str(exc.value)
