"""Provider-agnostic LLM client for scoring and analysis with automatic fallback."""
import asyncio
import json
import logging
import os
import re
from backend.models.db import SessionLocal, Setting

logger = logging.getLogger("jobnavigator.llm")


DEFAULT_PROVIDER = "claude_api"
DEFAULT_MODEL = "claude-sonnet-5"


def _get_setting(db, key, default=""):
    row = db.query(Setting).filter(Setting.key == key).first()
    return row.value if row and row.value else default


def resolve_llm_config(feature: str = "", db=None) -> dict:
    """Resolve the provider/model/api_key a feature dispatches with: `<feature>_llm_*` setting -> primary `llm_*` setting -> shipped default. Single source of truth for both dispatch and logging."""
    own_db = db is None
    if own_db:
        db = SessionLocal()
    try:
        prefix = f"{feature}_" if feature else ""
        provider = _get_setting(db, f"{prefix}llm_provider", "")
        model = _get_setting(db, f"{prefix}llm_model", "")
        api_key = _get_setting(db, f"{prefix}llm_api_key", "")
        if feature:
            provider = provider or _get_setting(db, "llm_provider", "")
            model = model or _get_setting(db, "llm_model", "")
            api_key = api_key or _get_setting(db, "llm_api_key", "")
        return {
            "provider": provider or DEFAULT_PROVIDER,
            "model": model or DEFAULT_MODEL,
            "api_key": api_key,
        }
    finally:
        if own_db:
            db.close()


async def call_llm(prompt: str, system: str, max_tokens: int = 1200,
                   cached_prefix: str | None = None,
                   provider: str | None = None, model: str | None = None,
                   api_key: str | None = None,
                   response_schema: dict | None = None,
                   schema_name: str = "response",
                   timeout: float | None = None,
                   temperature: float | None = None,
                   reasoning: bool | None = None) -> dict:
    """Route to the configured LLM provider with retry + automatic fallback; provider/model/api_key override the Primary for this call when given, else fall back to the llm_* settings. response_schema constrains the reply to that JSON Schema where the provider can enforce it (see _dispatch).

    On OpenRouter each attempt is pinned to a backend from the model's ladder (openrouter_ladder):
    a backend that cannot serve the request steps to the next rung immediately (no
    backoff — it is a different machine), while any other failure retries the same
    backend on the usual backoff.

    `timeout` is a per-attempt deadline in seconds. Exceeding it raises
    LLMTimeoutError and abandons the whole call: no further rungs, no fallback
    provider, because the caller is waiting on a person-facing request.
    """
    MAX_ATTEMPTS = 4
    BACKOFF_BASE = 2  # seconds: 2, 4, 8

    db = SessionLocal()
    try:
        primary = resolve_llm_config("", db=db)
        if provider is None:
            provider = primary["provider"]
        if model is None:
            model = primary["model"]
        if api_key is None:
            api_key = primary["api_key"]
        fallback_provider = _get_setting(db, "llm_fallback_provider", "")
        fallback_model = _get_setting(db, "llm_fallback_model", "")
        fb_api_key = _get_setting(db, "llm_fallback_api_key", "")
    finally:
        db.close()

    # Only claude_api supports explicit prompt caching — other providers get the
    # prefix concatenated into the prompt (no cache discount).
    caching = bool(cached_prefix) and provider == "claude_api"

    async def _try_pair(prov, mdl, key, label, caching_on):
        """MAX_ATTEMPTS against one provider/model, walking the OpenRouter backend ladder on availability failures. Returns the reply or raises the last error."""
        rung = 0
        last_err = None
        ladder = openrouter_ladder(mdl)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            on_ladder = prov == "openrouter"
            backend = ladder[rung] if on_ladder else None
            where = f", backend={backend or 'unpinned'}" if on_ladder else ""
            try:
                logger.info(f"LLM {label}: provider={prov}, model={mdl}, attempt={attempt}/{MAX_ATTEMPTS}{where}, caching={'on' if caching_on else 'off'}")
                call = _dispatch(prov, mdl, key, prompt, system, max_tokens, cached_prefix=cached_prefix,
                                 response_schema=response_schema, schema_name=schema_name, backend=backend,
                                 temperature=temperature, reasoning=reasoning)
                return await (asyncio.wait_for(call, timeout=timeout) if timeout else call)
            except (asyncio.TimeoutError, TimeoutError) as e:
                # Someone is holding a request open waiting for this; another rung
                # would only make them wait longer. Give up on the whole call.
                raise LLMTimeoutError(f"{prov}/{mdl} did not answer within {timeout:g}s") from e
            except Exception as e:
                last_err = e
                if isinstance(e, NonRetryableLLMError):
                    logger.warning(f"LLM {label} failed, not retrying: {e}")
                    break
                if on_ladder and _is_backend_unavailable(e) and rung + 1 < len(ladder):
                    rung += 1
                    nxt = ladder[rung] or "unpinned"
                    # A different machine — waiting first would buy nothing.
                    logger.warning(f"LLM {label} backend {backend} unavailable ({e}); stepping to {nxt}")
                    continue
                if attempt < MAX_ATTEMPTS:
                    wait = BACKOFF_BASE ** attempt  # 2, 4, 8
                    logger.warning(f"LLM {label} attempt {attempt}/{MAX_ATTEMPTS} failed: {e}, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    logger.warning(f"LLM {label} exhausted {MAX_ATTEMPTS} attempts: {e}")
        raise last_err if last_err else RuntimeError(f"LLM {label} made no attempt")

    # Try primary with retries
    last_primary_err = None
    try:
        res = await _try_pair(provider, model, api_key, "primary", caching)
        return {**res, "provider": provider, "model": model}
    except LLMTimeoutError:
        raise
    except Exception as e:
        last_primary_err = e

    # Try fallback with retries
    if fallback_provider and fallback_model:
        fb_caching = bool(cached_prefix) and fallback_provider == "claude_api"
        last_fallback_err = None
        try:
            res = await _try_pair(fallback_provider, fallback_model, fb_api_key, "fallback", fb_caching)
            # Report the pair that actually answered so the caller logs the
            # fallback, not the primary it never reached.
            return {**res, "provider": fallback_provider, "model": fallback_model}
        except LLMTimeoutError:
            raise
        except Exception as e:
            last_fallback_err = e

        raise RuntimeError(
            f"Both LLM providers failed after {MAX_ATTEMPTS} attempts each. "
            f"Primary ({provider}/{model}): {last_primary_err}. "
            f"Fallback ({fallback_provider}/{fallback_model}): {last_fallback_err}"
        )

    raise last_primary_err


async def call_email_llm(prompt: str, system: str, max_tokens: int = 150) -> dict:
    """Route to email-specific LLM provider. Returns {text, usage}."""
    cfg = resolve_llm_config("email")
    provider, model = cfg["provider"], cfg["model"]

    logger.info(f"Email LLM call: provider={provider}, model={model}, max_tokens={max_tokens}")
    res = await _dispatch(provider, model, cfg["api_key"], prompt, system, max_tokens)
    return {**res, "provider": provider, "model": model}


async def call_cv_tailor_llm(prompt: str, system: str, max_tokens: int = 3000,
                             response_schema: dict | None = None,
                             schema_name: str = "response",
                             temperature: float | None = None,
                             reasoning: bool | None = None) -> dict:
    """Route to the CV-tailoring provider through call_llm, so tailoring gets the same retries, OpenRouter backend ladder and fallback provider as résumé import. Returns {text, usage, provider, model} naming the pair that answered."""
    cfg = resolve_llm_config("cv_tailor")
    provider, model = cfg["provider"], cfg["model"]

    logger.info(f"CV tailor LLM call: provider={provider}, model={model}, max_tokens={max_tokens}")
    # No deadline: the run is in the background, so no one is waiting on a spinner.
    return await call_llm(prompt, system, max_tokens=max_tokens,
                          provider=provider, model=model, api_key=cfg["api_key"],
                          response_schema=response_schema, schema_name=schema_name,
                          temperature=temperature, reasoning=reasoning)


async def call_cover_letter_llm(prompt: str, system: str, max_tokens: int = 1500,
                                cached_prefix: str | None = None) -> dict:
    """Route to cover-letter-specific LLM provider; the resume + persona-preferences prefix caches on Claude API so regenerating in a different voice/length only pays for the JD suffix."""
    cfg = resolve_llm_config("cover_letter")
    provider, model = cfg["provider"], cfg["model"]

    logger.info(f"Cover-letter LLM call: provider={provider}, model={model}, max_tokens={max_tokens}, "
                f"caching={'on' if cached_prefix and provider == 'claude_api' else 'off'}")
    res = await _dispatch(provider, model, cfg["api_key"], prompt, system, max_tokens,
                          cached_prefix=cached_prefix)
    return {**res, "provider": provider, "model": model}


async def call_autofill_llm(prompt: str, system: str, max_tokens: int = 400,
                            cached_prefix: str | None = None) -> dict:
    """Route to autofill-specific LLM provider; the persona + qa_bank prefix caches on Claude API so regenerating with a different length/company only pays for the per-question suffix."""
    cfg = resolve_llm_config("autofill")
    provider, model = cfg["provider"], cfg["model"]

    logger.info(f"Autofill LLM call: provider={provider}, model={model}, max_tokens={max_tokens}, "
                f"caching={'on' if cached_prefix and provider == 'claude_api' else 'off'}")
    res = await _dispatch(provider, model, cfg["api_key"], prompt, system, max_tokens,
                          cached_prefix=cached_prefix)
    return {**res, "provider": provider, "model": model}


async def call_autofill_llm_stream(prompt: str, system: str, max_tokens: int = 400,
                                   cached_prefix: str | None = None):
    """Streaming version of call_autofill_llm; claude_api and openai/openrouter stream natively, other providers fall back to a single full-answer chunk."""
    cfg = resolve_llm_config("autofill")
    provider, model, api_key = cfg["provider"], cfg["model"], cfg["api_key"]

    if provider == "claude_api":
        async for c in _stream_claude(prompt, system, model, api_key, max_tokens, cached_prefix):
            yield c
        return
    combined = f"{cached_prefix}\n\n{prompt}" if cached_prefix else prompt
    if provider in ("openai", "openrouter"):
        base = OPENROUTER_BASE_URL if provider == "openrouter" else None
        async for c in _stream_openai(combined, system, model, api_key, max_tokens, base):
            yield c
        return
    # Non-streaming providers can't emit tokens as they generate, so simulate it:
    # fetch the full answer, then yield word-sized chunks so the draft still animates.
    import asyncio, re
    res = await _dispatch(provider, model, api_key, prompt, system, max_tokens, cached_prefix=cached_prefix)
    text = res.get("text", "") or ""
    for tok in re.findall(r"\S+\s*", text):
        yield tok
        await asyncio.sleep(0.012)


async def _stream_claude(prompt, system, model, api_key, max_tokens, cached_prefix):
    import anthropic, os
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    client = anthropic.AsyncAnthropic(api_key=key)
    if cached_prefix:
        content = [
            {"type": "text", "text": cached_prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt
    async with client.messages.stream(model=model, max_tokens=max_tokens, system=system,
                                      messages=[{"role": "user", "content": content}]) as stream:
        async for text in stream.text_stream:
            yield text


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter routes each request to one of ~22 backends serving the model, and which
# one it picks dominates latency: measured against the résumé schema, the same model
# ranged from 6s (together) to >180s (two timed out). The ladder pins the two that
# were fast on every sample, then falls through to unpinned so a bad day for both
# degrades to OpenRouter's own routing instead of failing the call.
# Re-measured on deepseek-v4.1-flash structuring a 3-page résumé (2026-09-23): together
# 30s, fireworks 42s, coreweave/fp8 33s, morph 38s, modal 69s, wafer 94s, atlas 85s,
# deepinfra/fp8 139s.
# Tags are per-model endpoint names; an unknown tag just fails and steps down.
OPENROUTER_BACKEND_LADDER = ("together", "fireworks", None)

# Backends differ per model, so a model measured the same way gets its own ladder.
# deepseek-v4-pro, structuring a 3-page résumé twice per backend (2026-09-22):
# deepinfra/fp8 42s and 13s, baidu/fp8 40s and 50s, against 98s unpinned; two backends
# ran past 8000 tokens and neither default pin serves the model.
OPENROUTER_MODEL_LADDERS = {
    "deepseek/deepseek-v4-pro": ("deepinfra/fp8", "baidu/fp8", None),
    # Checked 2026-09-22: no default pin serves these, so they start unpinned instead of
    # spending two rungs (and two of the four attempts) on "No endpoints found".
    # gemma-4-31b, tailoring the same résumé three times per backend (2026-09-23):
    # modelrun/fp4 answered in 6s flat, coreweave/fp4 in 26-111s, siliconflow/fp8 twice
    # returned a single role bullet. The fast one occasionally omits a project, which
    # build_tailored_resume now backfills, so speed comes first and coreweave follows.
    "google/gemma-4-31b-it": ("modelrun/fp4", "coreweave/fp4", None),
    "qwen/qwen3-30b-a3b-instruct-2507": (None,),
    "qwen/qwen3.8-27b": (None,),
    "openai/gpt-oss-safeguard-20b": (None,),
}


def openrouter_ladder(model: str) -> tuple:
    """The OpenRouter backend ladder for `model`: its own measured pins, else the default."""
    return OPENROUTER_MODEL_LADDERS.get(model, OPENROUTER_BACKEND_LADDER)


class BackendUnavailableError(RuntimeError):
    """This backend could not serve the request (saturated, down, or truncated the reply). Retrying it is pointless — step to the next rung."""


class LLMTimeoutError(RuntimeError):
    """A call exceeded the caller's deadline. Aborts the whole call_llm: no more rungs, no provider fallback."""


# OpenRouter relays a backend's own failure with the upstream status. These are the
# ones that mean "this machine can't take it", as opposed to "this request is wrong".
_UNAVAILABLE_STATUS = {429, 500, 502, 503, 504}


def _is_backend_unavailable(exc: Exception) -> bool:
    """True when the failure is the backend's availability rather than the request itself.

    A 429 relayed from a provider ("Provider returned error") means that backend is
    saturated; a 429 against the OpenRouter account itself means *we* are over a
    limit and stepping sideways would not help, so only the former steps down.
    """
    if isinstance(exc, BackendUnavailableError):
        return True
    status = getattr(exc, "status_code", None)
    if status in _UNAVAILABLE_STATUS:
        if status == 429:
            return "provider returned error" in str(exc).lower()
        return True
    # A pinned backend that doesn't serve this model (or can't honour the schema) leaves
    # OpenRouter no endpoint at all — "No endpoints found". Retrying that pin can never
    # succeed; the next rung, which may be unpinned, can.
    if status == 404 and "no endpoints found" in str(exc).lower():
        return True
    # No response at all: DNS, connection reset, read timeout.
    return type(exc).__name__ in ("APIConnectionError", "APITimeoutError", "ConnectError", "ReadTimeout")
# OpenRouter app attribution (lists the app on openrouter.ai/apps); ignored by other endpoints.
OPENROUTER_HEADERS = {"HTTP-Referer": "https://github.com/vesaias/JobNavigator", "X-Title": "JobNavigator"}


def _openai_client(api_key: str, base_url: str | None):
    from openai import AsyncOpenAI
    headers = OPENROUTER_HEADERS if base_url == OPENROUTER_BASE_URL else None
    return AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=headers)


async def _stream_openai(prompt, system, model, api_key, max_tokens, base_url=None):
    client = _openai_client(api_key, base_url)
    stream = await client.chat.completions.create(
        model=model, max_tokens=max_tokens, stream=True,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta


def _json_schema_format(schema: dict, name: str) -> dict:
    """An OpenAI-style response_format for `schema`; strict mode is what makes it a constraint rather than a hint."""
    return {"type": "json_schema",
            "json_schema": {"name": name, "strict": True, "schema": schema}}


def _openrouter_routing(backend: str | None, schema: dict | None) -> dict | None:
    """OpenRouter's `provider` routing block: pin this rung's backend, or, on the unpinned rung, at least keep the routing to backends that honour the schema (10 of the 22 serving our model do not)."""
    if backend:
        return {"provider": {"order": [backend], "allow_fallbacks": False}}
    if schema:
        return {"provider": {"require_parameters": True}}
    return None


def _schema_in_prompt(prompt: str, schema: dict | None) -> str:
    """The last-resort fallback for the CLI providers, which drive a coding agent and take no response format: the schema goes in the prompt, where it is a request rather than a constraint."""
    if not schema:
        return prompt
    return (f"{prompt}\n\nReturn one JSON object and nothing else. It must validate "
            f"against this JSON Schema:\n{json.dumps(schema)}")


async def _dispatch(provider: str, model: str, api_key: str,
                    prompt: str, system: str, max_tokens: int,
                    cached_prefix: str | None = None,
                    response_schema: dict | None = None,
                    schema_name: str = "response",
                    backend: str | None = None,
                    temperature: float | None = None,
                    reasoning: bool | None = None) -> dict:
    """Route to the correct provider; only `claude_api` supports prompt caching, others get cached_prefix concatenated into the prompt (no cache discount).

    response_schema is enforced where the API can: an OpenAI-style json_schema
    response format (openai/openrouter/lmstudio), Ollama's `format`, or a forced
    tool call on claude_api. The three CLI providers have no such knob, so they
    fall back to carrying the schema in the prompt.

    `backend` pins the OpenRouter endpoint (see openrouter_ladder); it means
    nothing to the other providers, which each have exactly one backend.

    `temperature` is passed to the providers that take one; the three CLI providers
    have no such knob and run at their own default. `reasoning=False` asks OpenRouter to
    turn a reasoning model's thinking off: on the résumé schema that cut deepseek-v4.1-flash
    from 10k output tokens in 30s to 2.7k in 8s, with the same answer.
    """
    if provider == "claude_api":
        return await _call_claude_api(prompt, system, model, api_key, max_tokens, cached_prefix=cached_prefix,
                                      response_schema=response_schema, schema_name=schema_name,
                                      temperature=temperature)
    combined = f"{cached_prefix}\n\n{prompt}" if cached_prefix else prompt
    if provider == "claude_code":
        return await _call_claude_code(combined, system, model, max_tokens,
                                       response_schema, reasoning)
    elif provider == "codex_cli":
        return await _call_codex_cli(combined, system, model, max_tokens, response_schema)
    elif provider == "antigravity_cli":
        return await _call_antigravity_cli(_schema_in_prompt(combined, response_schema), system, model, max_tokens)
    elif provider == "openai":
        return await _call_openai(combined, system, model, api_key, max_tokens,
                                  response_schema=response_schema, schema_name=schema_name,
                                  temperature=temperature)
    elif provider == "openrouter":
        # OpenRouter is OpenAI-API-compatible — same client, different base URL.
        # One key reaches every vendor's models (model slug is vendor-prefixed).
        routing = _openrouter_routing(backend, response_schema) or {}
        if reasoning is False:
            routing = {**routing, "reasoning": {"enabled": False}}
        return await _call_openai(combined, system, model, api_key, max_tokens,
                                  base_url=OPENROUTER_BASE_URL,
                                  extra_body=routing or None,
                                  response_schema=response_schema, schema_name=schema_name,
                                  temperature=temperature)
    elif provider == "lmstudio":
        # LM Studio serves an OpenAI-compatible API on :1234; no key (client wants a non-empty string).
        # Override with LMSTUDIO_BASE_URL when the backend is containerized (e.g. http://host.docker.internal:1234/v1).
        base = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
        # Thinking models (e.g. Qwen3) burn the whole max_tokens budget on reasoning and
        # return empty content; disable thinking for these structured-output calls.
        # LM Studio's OpenAI-compatible endpoint takes reasoning_effort="none" to turn it off.
        return await _call_openai(combined, system, model, api_key or "lm-studio", max_tokens,
                                  base_url=base,
                                  extra_body={"reasoning_effort": "none"},
                                  response_schema=response_schema, schema_name=schema_name,
                                  temperature=temperature)
    elif provider == "ollama":
        return await _call_ollama(combined, system, model, max_tokens,
                                  response_schema=response_schema, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def _call_claude_api(prompt: str, system: str, model: str, api_key: str,
                           max_tokens: int, cached_prefix: str | None = None,
                           response_schema: dict | None = None, schema_name: str = "response",
                           temperature: float | None = None) -> dict:
    """Call Claude via the Anthropic SDK; cached_prefix is sent as a separate cache_control block for ~10x cheaper reuse, but is ignored below the 1024-token (Sonnet/Opus) minimum.

    The Messages API has no response_format, so a response_schema is enforced the
    way Anthropic intends: a single tool whose input_schema is that schema, with
    tool_choice forcing it. The tool's arguments come back as the reply text, so
    callers keep parsing a JSON string either way.
    """
    import anthropic
    key = api_key or __import__('os').getenv("ANTHROPIC_API_KEY", "")
    client = anthropic.AsyncAnthropic(api_key=key)

    if cached_prefix:
        content = [
            {"type": "text", "text": cached_prefix, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt  # plain string — no cache_control

    schema_kwargs = {}
    if response_schema:
        schema_kwargs = {
            "tools": [{"name": schema_name,
                       "description": f"Return the {schema_name} as structured data.",
                       "input_schema": response_schema}],
            "tool_choice": {"type": "tool", "name": schema_name},
        }

    if temperature is not None:
        schema_kwargs["temperature"] = temperature
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        **schema_kwargs,
    )

    if response_schema:
        # The forced call is the whole reply; hand its arguments back as JSON text.
        block = next((b for b in response.content if getattr(b, "type", "") == "tool_use"), None)
        if block is None:
            raise RuntimeError(f"Claude returned no {schema_name} tool call")
        text = json.dumps(block.input)
    else:
        text = response.content[0].text.strip()

    # Extract usage — cache_* attributes may be absent on older SDK versions or non-cached calls
    usage = response.usage
    return {
        "text": text,
        "usage": {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        },
    }


# Whether the installed `claude` supports --json-schema. None until probed; the answer
# cannot change under a running process, so it is asked once.
_CLAUDE_JSON_SCHEMA: bool | None = None


async def _claude_supports_json_schema() -> bool:
    """Whether this `claude` can constrain its reply to a schema rather than be asked to.

    The flag arrived after this adapter was written. An older CLI errors on an unknown
    option, which would turn every call into a failure, so support is checked once and
    the prompt-level request stays as the fallback.
    """
    global _CLAUDE_JSON_SCHEMA
    if _CLAUDE_JSON_SCHEMA is None:
        try:
            rc, out, _ = await _run_cli(["claude", "--help"], b"", timeout=30)
            _CLAUDE_JSON_SCHEMA = rc == 0 and b"--json-schema" in out
        except Exception as e:
            logger.warning(f"Could not ask the claude CLI about --json-schema ({e}); "
                           "falling back to the schema in the prompt")
            _CLAUDE_JSON_SCHEMA = False
        logger.info(f"Claude Code: structured output via --json-schema is "
                    f"{'available' if _CLAUDE_JSON_SCHEMA else 'NOT available'}")
    return _CLAUDE_JSON_SCHEMA


async def _call_claude_code(prompt: str, system: str, model: str, max_tokens: int,
                             response_schema: dict | None = None,
                             reasoning: bool | None = None) -> dict:
    """Call Claude via claude CLI subprocess. Returns {text, usage}. Caching not supported."""
    import os
    import json as _json
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd.extend(["--model", model])

    cmd.extend(["--effort", {False: "low", True: "medium"}.get(reasoning, "medium")])

    if response_schema and await _claude_supports_json_schema():
        cmd.extend(["--json-schema", _json.dumps(response_schema)])
        full_prompt = f"{system}\n\n{prompt}"
    else:
        full_prompt = f"{system}\n\n{_schema_in_prompt(prompt, response_schema)}"

    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    rc, stdout, stderr = await _run_cli(cmd, full_prompt.encode(), env=env)

    if rc != 0:
        error = stderr.decode(errors="replace").strip()
        raise RuntimeError(f"claude-code subprocess failed (rc={rc}): {error}")

    raw = stdout.decode().strip()
    try:
        data = _json.loads(raw)
        text = data.get("result", raw)
    except _json.JSONDecodeError:
        data, text = None, raw

    # `claude -p --output-format json` reports a refusal, an overload or a hit
    # quota as a rc=0 envelope with is_error set. Returning its text as if it
    # were the answer buried the reason in whatever the caller did next.
    if isinstance(data, dict) and (data.get("is_error") or data.get("subtype") not in (None, "success")):
        reason = (str(text or "").strip() or str(data.get("error") or "").strip()
                  or str(data.get("subtype") or "").strip() or "no reason given")
        if _QUOTA_RE.search(reason):
            raise NonRetryableLLMError(f"Claude Code usage limit reached: {reason[:300]}")
        raise RuntimeError(f"Claude Code reported an error: {reason[:300]}")
    if not str(text or "").strip():
        # Retryable on purpose: an empty completion is a bad minute, not a bad setup.
        raise RuntimeError("Claude Code returned an empty reply")

    return {
        "text": text.strip(),
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0},
    }


class CLITimeoutError(RuntimeError):
    """A CLI that outran its wall clock, carrying whatever it printed before the kill."""

    def __init__(self, message: str, stdout: bytes = b"", stderr: bytes = b""):
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class NonRetryableLLMError(RuntimeError):
    """A failure a retry cannot fix (not logged in, usage limit); call_llm goes straight to the fallback."""


CLI_TIMEOUT = 300   # seconds one subscription-CLI completion may take before it is killed
# codex exec processes share one auth.json and rewrite it on token refresh; OpenAI's docs say not
# to run many against the same file, so at most two at a time whatever the scoring limiter allows.
_codex_gate = asyncio.Semaphore(2)
_CODEX_LOGIN_HINT = "run `docker compose exec backend codex login --device-auth`"
_QUOTA_RE = re.compile(r"usage limit|rate limit|quota|too many requests|\b429\b", re.I)
# agy refreshes one shared OAuth token file, so hold the same limit codex gets.
_agy_gate = asyncio.Semaphore(2)
_AGY_LOGIN_HINT = "run `docker compose exec -it backend agy` and sign in"
_AGY_AUTH_RE = re.compile(r"authentication (required|failed)|not logged in|sign in", re.I)
_AGY_STATE = "~/.gemini/antigravity-cli"
# The deny list the image ships. It lives outside the state directory because agy rewrites
# settings.json on every start, and a read-only copy there only makes that write fail.
_AGY_SETTINGS_SOURCE = "/opt/antigravity-settings.json"


async def _run_cli(cmd: list[str], stdin: bytes, env: dict | None = None, timeout: float = CLI_TIMEOUT,
                   cwd: str | None = None):
    """Run a CLI to completion with a hard timeout; returns (rc, stdout, stderr)."""
    process = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env, cwd=cwd,
    )
    reader = asyncio.ensure_future(process.communicate(input=stdin))
    try:
        # shield, so a timeout does not cancel the read: after the kill the same task
        # hands back what the process managed to print, which is how a caller can still
        # clean up after a call that hung.
        stdout, stderr = await asyncio.wait_for(asyncio.shield(reader), timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        partial = (b"", b"")
        try:
            partial = await asyncio.wait_for(reader, 5) or partial
        except (asyncio.TimeoutError, OSError, ValueError, TypeError):
            pass
        raise CLITimeoutError(f"{cmd[0]} timed out after {int(timeout)}s", *partial)
    return process.returncode, stdout, stderr


async def _call_codex_cli(prompt: str, system: str, model: str, max_tokens: int,
                          response_schema: dict | None = None) -> dict:
    """Call Codex CLI using its existing ChatGPT login; run in an empty read-only workspace."""
    import json as _json
    import os
    import tempfile

    # like ANTHROPIC_API_KEY for claude_code: the plan pays, never an API key that happens to be set
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY")}
    async with _codex_gate:
        # `codex exec` without a login retries for ~15 s and then prints a 401 storm; the status
        # check answers in milliseconds and names the fix
        rc, _, _ = await _run_cli(["codex", "login", "status"], b"", env=env, timeout=30)
        if rc != 0:
            raise NonRetryableLLMError(f"Codex CLI is not logged in — {_CODEX_LOGIN_HINT}")

        full_prompt = f"{system}\n\n{prompt}"
        with tempfile.TemporaryDirectory(prefix="jobnavigator-codex-") as workdir:
            # `--output-schema` constrains the final message the way an API's response_format
            # does, so the schema stops being a request in the prompt that the model may
            # answer around. It takes a path rather than a string, and the ephemeral
            # workspace is already here, so the file lives and dies with the call.
            schema_path = ""
            if response_schema:
                schema_path = os.path.join(workdir, "response_schema.json")
                with open(schema_path, "w", encoding="utf-8") as fh:
                    _json.dump(response_schema, fh)
            else:
                full_prompt = f"{system}\n\n{_schema_in_prompt(prompt, None)}"
            cmd = [
                "codex", "exec", "--ephemeral", "--ignore-user-config", "--ignore-rules",
                "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
                "-c", "web_search=disabled", "-c", "history.persistence=none",
                "-c", "check_for_update_on_startup=false",
                "--json", "-C", workdir,
            ]
            if model:
                cmd.extend(["--model", model])
            if schema_path:
                cmd.extend(["--output-schema", schema_path])
            cmd.append("-")
            rc, stdout, stderr = await _run_cli(cmd, full_prompt.encode(), env=env)

    raw = stdout.decode(errors="replace").strip()
    text = ""
    failed = ""      # turn.failed carries the one readable reason (401, usage limit, model unknown)
    last_error = ""  # transport-level `error` events, the fallback reason when the turn never fails
    usage = {"input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0}
    for line in raw.splitlines():
        try:
            event = _json.loads(line)
        except _json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                text = item["text"]
        elif kind == "turn.completed":
            token_usage = event.get("usage") or {}
            usage.update({
                "input_tokens": token_usage.get("input_tokens", 0) or 0,
                "output_tokens": token_usage.get("output_tokens", 0) or 0,
                "cache_read_tokens": token_usage.get("cached_input_tokens", 0) or 0,
            })
        elif kind == "turn.failed":
            failed = str((event.get("error") or {}).get("message") or "")
        elif kind == "error":
            last_error = str(event.get("message") or "")

    if failed or rc != 0:
        err_lines = stderr.decode(errors="replace").strip().splitlines()
        reason = failed or last_error or (err_lines[-1] if err_lines else "no output")
        if "401" in reason or "Unauthorized" in reason:
            raise NonRetryableLLMError(f"Codex CLI is not logged in — {_CODEX_LOGIN_HINT} ({reason})")
        if _QUOTA_RE.search(reason):
            raise NonRetryableLLMError(f"Codex usage limit reached: {reason}")
        raise RuntimeError(f"codex exec failed (rc={rc}): {reason}")
    if not text:
        raise RuntimeError("codex exec completed without an agent response")
    return {"text": text.strip(), "usage": usage}


def _agy_events(stdout: bytes):
    """Every NDJSON event agy printed, skipping any line that is not one."""
    import json as _json
    for line in stdout.decode(errors="replace").splitlines():
        try:
            yield _json.loads(line)
        except _json.JSONDecodeError:
            continue


def _agy_conversation_id(stdout: bytes) -> str:
    """The id agy gave this call. The init event carries it before any answer, which is all
    a timed-out call leaves behind."""
    for event in _agy_events(stdout):
        cid = event.get("conversation_id") or (event.get("result") or {}).get("conversation_id")
        if cid:
            return str(cid)
    return ""


def _agy_install_settings() -> list[str]:
    """Put the shipped deny list where agy reads it, and answer with the rules it must honour.
    agy rewrites settings.json as it starts, so the file has to be writable: pinning a
    read-only copy there makes the write fail and drops the CLI onto its defaults."""
    import json as _json
    import shutil
    import os
    target = os.path.join(os.path.expanduser(_AGY_STATE), "settings.json")
    try:
        with open(_AGY_SETTINGS_SOURCE, encoding="utf-8") as fh:
            source = fh.read()
    except OSError as e:
        raise NonRetryableLLMError(
            f"Antigravity deny list is missing at {_AGY_SETTINGS_SOURCE} ({e}); "
            "the agent would run unconstrained, so the call is refused")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        current = open(target, encoding="utf-8").read()
    except OSError:
        current = ""
    if current != source:
        tmp = f"{target}.jobnavigator.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(source)
        shutil.move(tmp, target)
    return list(_json.loads(source).get("permissions", {}).get("deny", []))


def _agy_assert_denied(log_path: str, rules: list[str]) -> None:
    """Fail the call unless agy's own log says it loaded every deny rule.
    Without this the CLI can silently fall back to defaults and hand the agent
    `run_command` and `write_file` back, and nothing in the answer would show it."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            loaded = [ln for ln in fh if "CLI settings initialized" in ln]
    except OSError:
        loaded = []
    if not loaded:
        raise NonRetryableLLMError(
            "Antigravity CLI never reported its permissions, so the deny list cannot be "
            "confirmed; the call is refused rather than run unconstrained")
    line = loaded[-1]
    missing = [r for r in rules if r not in line]
    if missing:
        raise NonRetryableLLMError(
            f"Antigravity CLI started without the deny rules {missing}; "
            "it would be free to read files and run commands, so the call is refused")


def _agy_discard(conversation_id: str) -> None:
    """Delete the per-call transcript agy writes. It holds the whole prompt — the resume text —
    and nothing reads it back, so it is disk growth and a copy of user data we do not want."""
    import glob
    import os
    import shutil
    home = os.path.expanduser(_AGY_STATE)
    for path in (*glob.glob(f"{home}/conversations/{conversation_id}.db*"),
                 f"{home}/presence/{conversation_id}.lock",
                 f"{home}/annotations/{conversation_id}.pbtxt",
                 f"{home}/brain/{conversation_id}"):
        try:
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
        except OSError:
            pass   # a leftover file is untidy, never a reason to fail a scored job


async def _call_antigravity_cli(prompt: str, system: str, model: str, max_tokens: int) -> dict:
    """Call Antigravity CLI on its Google subscription. The prompt rides on stdin as one NDJSON
    line because `-p` reads argv and argv caps near 128 KB. `max_tokens` is ignored: the CLI
    exposes no output cap. Tool use is blocked by a deny list installed before each call and
    confirmed from agy's own log afterwards — `--sandbox` restricts the terminal only, never
    the file tools."""
    import json as _json
    import os
    import tempfile

    # like ANTHROPIC_API_KEY for claude_code: the plan pays, never a key that happens to be set
    env = {k: v for k, v in os.environ.items()
           if k not in ("GEMINI_API_KEY", "GOOGLE_API_KEY",
                        "GOOGLE_APPLICATION_CREDENTIALS", "AGY_ADC_AUTH")}
    message = _json.dumps({"event": "user",
                           "message": {"role": "user", "content": f"{system}\n\n{prompt}"}})

    denied = _agy_install_settings()
    stdout = b""
    try:
        async with _agy_gate:
            with tempfile.TemporaryDirectory(prefix="jobnavigator-agy-") as workdir:
                # The log goes in the temp directory: it holds the permissions line this
                # call is checked against, and ~16 KB per call has no business surviving.
                log_path = os.path.join(workdir, "cli.log")
                # `-p` always swallows the next token, so it goes last with an attached empty value.
                cmd = ["agy", "--input-format", "stream-json", "--output-format", "stream-json",
                       "--print-timeout", f"{int(CLI_TIMEOUT)}s", "--log-file", log_path]
                if model:
                    cmd.extend(["--model", model])
                cmd.append("-p=")
                rc, stdout, stderr = await _run_cli(cmd, (message + "\n").encode(),
                                                    env=env, cwd=workdir)
                _agy_assert_denied(log_path, denied)
    except CLITimeoutError as e:
        stdout = e.stdout
        raise
    finally:
        # A timeout kills agy before the result event, so the id comes from the init event.
        # The transcript has to go exactly in that case too — it holds the whole prompt.
        conversation_id = _agy_conversation_id(stdout)
        if conversation_id:
            _agy_discard(conversation_id)

    result = {}
    for event in _agy_events(stdout):
        if event.get("event") == "result":
            result = event.get("result") or {}

    if result.get("status") != "SUCCESS" or rc != 0:
        err_lines = stderr.decode(errors="replace").strip().splitlines()
        reason = str(result.get("error") or "") or (err_lines[-1] if err_lines else "no output")
        if _AGY_AUTH_RE.search(reason):
            raise NonRetryableLLMError(f"Antigravity CLI is not logged in — {_AGY_LOGIN_HINT} ({reason})")
        if _QUOTA_RE.search(reason):
            raise NonRetryableLLMError(f"Antigravity usage limit reached: {reason}")
        raise RuntimeError(f"agy failed (rc={rc}): {reason}")

    text = str(result.get("response") or "").strip()
    if not text:
        raise RuntimeError("agy completed without a response")
    usage = result.get("usage") or {}
    return {
        "text": text,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "cache_read_tokens": usage.get("cache_read_tokens", 0) or 0,
            "cache_write_tokens": 0,
        },
    }


async def _call_openai(prompt: str, system: str, model: str, api_key: str, max_tokens: int,
                       base_url: str | None = None, extra_body: dict | None = None,
                       response_schema: dict | None = None, schema_name: str = "response",
                       temperature: float | None = None) -> dict:
    """Call the OpenAI API, or any OpenAI-compatible endpoint via base_url; response_schema is sent as a strict json_schema response format."""
    client = _openai_client(api_key, base_url)  # base_url=None → OpenAI default
    kwargs = {"extra_body": extra_body} if extra_body else {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if response_schema:
        kwargs["response_format"] = _json_schema_format(response_schema, schema_name)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        **kwargs,
    )
    choice = response.choices[0]
    # A reply cut off at max_tokens is unusable — under a schema it is a JSON object
    # with no closing braces. Backends differ wildly in how much they pad (one emitted
    # 4875 tokens where another used 2030 for the same input), so this is the backend's
    # problem and worth stepping the ladder for, not a parse error to show the user.
    if choice.finish_reason == "length":
        raise BackendUnavailableError(
            f"reply truncated at max_tokens={max_tokens} (finish_reason=length)")

    usage = response.usage
    return {
        "text": choice.message.content.strip(),
        "usage": {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        },
    }


async def _call_ollama(prompt: str, system: str, model: str, max_tokens: int,
                       response_schema: dict | None = None,
                       temperature: float | None = None) -> dict:
    """Call local Ollama instance. Returns {text, usage}. Ollama takes a JSON Schema directly as `format`."""
    import httpx
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"num_predict": max_tokens,
                    **({"temperature": temperature} if temperature is not None else {})},
    }
    if response_schema:
        payload["format"] = response_schema
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "http://localhost:11434/api/generate",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return {
        "text": data["response"].strip(),
        "usage": {
            "input_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        },
    }
