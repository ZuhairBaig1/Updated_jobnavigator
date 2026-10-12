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


def test_the_tokens_the_cli_reports_are_passed_on():
    from backend.analyzer.llm_client import _claude_code_usage
    envelope = {"result": "{}", "usage": {"input_tokens": 2, "cache_creation_input_tokens": 297,
                                          "cache_read_input_tokens": 9774, "output_tokens": 4356}}

    assert _claude_code_usage(envelope) == {"input_tokens": 2, "output_tokens": 4356, "thinking_tokens": 0,
                                            "cache_read_tokens": 9774, "cache_write_tokens": 297}


def test_a_reply_with_no_usage_counts_as_no_tokens():
    from backend.analyzer.llm_client import _claude_code_usage
    nothing = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "thinking_tokens": 0}

    assert _claude_code_usage(None) == _claude_code_usage({}) == _claude_code_usage({"usage": "n/a"}) == nothing


@pytest.mark.asyncio
async def test_the_cli_is_given_room_for_thinking_and_a_long_answer(monkeypatch):
    import json
    import backend.analyzer.llm_client as L
    seen = {}

    async def fake_run_cli(cmd, stdin, env=None, **options):
        seen.update(env or {})
        return 0, json.dumps({"result": "ok"}).encode(), b""
    monkeypatch.setattr(L, "_run_cli", fake_run_cli)

    await L._call_claude_code("prompt", "system", "claude-haiku-5-5", 16000)

    assert seen["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"


def test_thinking_is_reported_as_part_of_the_output():
    from backend.analyzer.llm_client import _claude_code_usage
    envelope = {"usage": {"input_tokens": 2, "output_tokens": 4757, "output_tokens_details": {"thinking_tokens": 3666}}}

    used = _claude_code_usage(envelope)

    assert (used["output_tokens"], used["thinking_tokens"]) == (4757, 3666)


# ── The Claude plan's usage limits ─────────────────────────────────────────────

LIMITS_EVENT = {"type": "rate_limit_event", "rate_limit_info": {
    "status": "allowed", "unifiedWindows": {"five_hour": {"utilization": 0.42, "resetsAt": 1791837600},
                                            "seven_day": {"utilization": 0.62, "resetsAt": 1791914400}}}}


def test_the_limits_are_read_from_the_event_the_cli_sends_with_a_call():
    from backend.analyzer.llm_client import _claude_code_limits_in
    stream = "\n".join([_json.dumps({"type": "system", "subtype": "init"}), "not json", _json.dumps(LIMITS_EVENT)])

    assert _claude_code_limits_in(stream) == {"status": "allowed",
                                              "five_hour": {"used": 0.42, "resets_at": 1791837600},
                                              "seven_day": {"used": 0.62, "resets_at": 1791914400}}


def test_a_stream_with_no_limits_event_gives_none():
    from backend.analyzer.llm_client import _claude_code_limits_in

    assert _claude_code_limits_in(_json.dumps({"type": "result"})) is None
    assert _claude_code_limits_in("") is None


@pytest.mark.asyncio
async def test_the_limits_are_asked_for_once_and_kept_for_a_minute(monkeypatch):
    import backend.analyzer.llm_client as L
    asked = []

    async def fake_run_cli(cmd, stdin, **options):
        asked.append(cmd)
        return 0, _json.dumps(LIMITS_EVENT).encode(), b""
    monkeypatch.setattr(L, "_run_cli", fake_run_cli)
    monkeypatch.setattr(L, "_claude_code_limits_read", {"at": 0.0, "limits": None})

    first = await L.claude_code_limits()
    second = await L.claude_code_limits()

    assert first["five_hour"]["used"] == 0.42 and second == first
    assert len(asked) == 1
    assert "claude-haiku-5-5" in asked[0]


def test_the_usage_endpoint_gives_the_limits_and_the_latest_calls_with_full_token_counts(api_client, test_db, monkeypatch):
    import backend.analyzer.llm_client as L
    from backend.models.db import LlmCallLog, Setting

    async def limits():
        return {"status": "allowed", "five_hour": {"used": 0.42, "resets_at": 1}, "seven_day": {"used": 0.62, "resets_at": 2}}
    monkeypatch.setattr(L, "claude_code_limits", limits)
    test_db.add(Setting(key="dashboard_api_key", value=""))
    test_db.add(LlmCallLog(purpose="tailor", provider="claude_code", model="claude-haiku-5-5", input_tokens=2,
                           cache_read_tokens=13863, cache_write_tokens=301, output_tokens=31554, duration_ms=111000, success=True))
    test_db.add(LlmCallLog(purpose="tailor", provider="openrouter", model="google/gemma-4-31b-it", input_tokens=7000,
                           output_tokens=3000, success=True))
    test_db.commit()

    reply = api_client.get("/api/llm/claude-code-usage")

    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["limits"]["five_hour"]["used"] == 0.42
    assert body["calls"] == [{"job": "Tailor", "model": "claude-haiku-5-5", "at": body["calls"][0]["at"],
                              "input_tokens": 14166, "output_tokens": 31554, "seconds": 111.0, "succeeded": True}]
    assert body["added_per_call"] == {"always": 340, "for_a_reply_format": 500}
