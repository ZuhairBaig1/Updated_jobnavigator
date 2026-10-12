"""LLM helper endpoints — live model catalogs for the Settings model picker; OpenRouter's is public, OpenAI/Anthropic require a resolved API key, and results are cached ~1h per provider."""
import os
import re
import time
import logging
import httpx
from fastapi import APIRouter, HTTPException
from backend.models.db import SessionLocal, Setting

logger = logging.getLogger("jobnavigator.llm")
router = APIRouter(prefix="/llm", tags=["settings"])

_CACHE_TTL = 3600  # 1 hour
_cache = {}  # provider -> {"at": float, "models": list}

# Provider setting slots to scan for a usable API key, in priority order.
_KEY_SLOTS = [
    ("llm_provider", "llm_api_key"),
    ("llm_fallback_provider", "llm_fallback_api_key"),
    ("cv_tailor_llm_provider", "cv_tailor_llm_api_key"),
    ("cover_letter_llm_provider", "cover_letter_llm_api_key"),
    ("autofill_llm_provider", "autofill_llm_api_key"),
    ("email_llm_provider", "email_llm_api_key"),
]
_ENV_KEY = {"openai": "OPENAI_API_KEY", "claude_api": "ANTHROPIC_API_KEY"}


def _get(db, key: str) -> str:
    row = db.query(Setting).filter(Setting.key == key).first()
    return (row.value or "") if row else ""


def _resolve_key(db, provider: str) -> str:
    """Find an API key already configured for `provider` in any slot, else env."""
    for pkey, kkey in _KEY_SLOTS:
        if _get(db, pkey) == provider:
            val = _get(db, kkey)
            if val:
                return val
    return os.getenv(_ENV_KEY.get(provider, ""), "")


async def _fetch_openrouter() -> list:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get("https://openrouter.ai/api/v1/models")
        resp.raise_for_status()
        data = resp.json().get("data", [])
    out = []
    for m in data:
        pricing = m.get("pricing") or {}
        out.append({
            "id": m.get("id"),
            "name": m.get("name") or m.get("id"),
            "context_length": m.get("context_length"),
            "prompt_price": pricing.get("prompt"),
            "completion_price": pricing.get("completion"),
        })
    return out


async def _fetch_openai(key: str) -> list:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get("https://api.openai.com/v1/models",
                                headers={"Authorization": f"Bearer {key}"})
        resp.raise_for_status()
        data = resp.json().get("data", [])
    out = []
    for m in data:
        mid = m.get("id", "")
        # Keep chat-capable families; skip embeddings/audio/image/moderation.
        if mid.startswith("gpt") or re.match(r"^(o\d|chatgpt)", mid):
            out.append({"id": mid, "name": mid})
    return out


async def _fetch_anthropic(key: str) -> list:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get("https://api.anthropic.com/v1/models?limit=1000",
                                headers={"x-api-key": key,
                                         "anthropic-version": "2023-06-01"})
        resp.raise_for_status()
        data = resp.json().get("data", [])
    return [{"id": m.get("id"), "name": m.get("display_name") or m.get("id")} for m in data]


@router.get("/models")
async def list_models(provider: str = "openrouter"):
    """Live model catalog for a provider, cached ~1h; openrouter needs no key, openai/claude_api use a configured key or env fallback (400 if none, 502 if the provider rejects it)."""
    now = time.time()
    hit = _cache.get(provider)
    if hit and (now - hit["at"]) < _CACHE_TTL:
        return {"models": hit["models"], "cached": True}

    db = SessionLocal()
    try:
        if provider == "openrouter":
            fetch = _fetch_openrouter()
        elif provider in ("openai", "claude_api", "claude_code"):
            # Claude Code (subscription) serves the same models as Claude API, so it
            # shares the Anthropic catalog — resolved via any configured Anthropic key.
            key_provider = "claude_api" if provider == "claude_code" else provider
            key = _resolve_key(db, key_provider)
            if not key:
                label = "OpenAI" if provider == "openai" else "Anthropic"
                raise HTTPException(400, f"No {label} API key configured — set one in Settings first.")
            fetch = _fetch_openai(key) if provider == "openai" else _fetch_anthropic(key)
        else:
            raise HTTPException(400, f"Live model search is not supported for provider '{provider}'.")
    finally:
        db.close()

    try:
        models = await fetch
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            raise HTTPException(502, "The provider rejected the configured API key.")
        logger.warning("model fetch failed (%s): %s", provider, e)
        if hit:
            return {"models": hit["models"], "cached": True, "stale": True}
        raise HTTPException(502, "Could not reach the provider's model catalog.")
    except Exception as e:
        logger.warning("model fetch failed (%s): %s", provider, e)
        if hit:
            return {"models": hit["models"], "cached": True, "stale": True}
        raise HTTPException(502, "Could not reach the provider's model catalog.")

    models = [m for m in models if m.get("id")]
    models.sort(key=lambda m: m["id"])
    _cache[provider] = {"at": now, "models": models}
    return {"models": models, "cached": False}


# ── The Claude plan: its usage limits, and what each call on it used ───────────────────

MOST_CALLS_LISTED = 50
JOB_OF_A_CALL = {"pdf": "Import", "tailor": "Tailor", "ats_score": "ATS score"}


@router.get("/claude-code-usage")
async def claude_code_usage(calls: int = 15):
    """How much of the Claude plan's 5-hour and 7-day limits is used, and the tokens of the latest Claude Code calls.

    Every count is what the CLI reported for that call, so it includes what the CLI adds to a prompt and the
    model's thinking. `added_per_call` says how much of the input that addition is.
    """
    from backend.analyzer import llm_client
    from backend.models.db import LlmCallLog

    limits = await llm_client.claude_code_limits()
    db = SessionLocal()
    try:
        rows = (db.query(LlmCallLog).filter(LlmCallLog.provider == "claude_code")
                .order_by(LlmCallLog.created_at.desc()).limit(max(1, min(calls, MOST_CALLS_LISTED))).all())
        latest = [{"job": JOB_OF_A_CALL.get(row.purpose, row.purpose), "model": row.model,
                   "at": row.created_at.isoformat() if row.created_at else None,
                   "input_tokens": (row.input_tokens or 0) + (row.cache_read_tokens or 0) + (row.cache_write_tokens or 0),
                   "output_tokens": row.output_tokens or 0,
                   "seconds": round((row.duration_ms or 0) / 1000, 1), "succeeded": bool(row.success)}
                  for row in rows]
    finally:
        db.close()
    return {"limits": limits, "calls": latest,
            "added_per_call": {"always": llm_client.CLAUDE_CODE_TOKENS_ADDED_PER_CALL,
                               "for_a_reply_format": llm_client.CLAUDE_CODE_TOKENS_ADDED_FOR_A_REPLY_FORMAT}}
