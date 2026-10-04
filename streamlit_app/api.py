"""HTTP client for the JobNavigator backend.

This app stores nothing of its own. Every read and write goes to the same FastAPI
endpoints the React frontend calls, so the two frontends always see one database —
which is the point: the résumé you tailor here shows up there, and the other way round.

Configuration comes from Streamlit secrets first, then the environment, so the same
code runs locally against a compose stack and on Streamlit Cloud against a deployed
backend without edits:

    API_BASE_URL   where the backend lives, e.g. https://jobnavigator.vercel.app
    API_KEY        the dashboard key, only needed once one is set in Settings
"""
import os
import re
from urllib.parse import unquote

import requests
import streamlit as st

# Tailoring is a background run; the POST returns immediately and the rest is polling,
# so no single request needs a long timeout.
TIMEOUT = 30
PREVIEW_TIMEOUT = 60
# Importing a PDF is not a background run: the endpoint extracts the text and waits on the
# LLM that structures it before replying, so this one request carries the whole wait.
IMPORT_TIMEOUT = 300
# A PDF is rendered by headless Chromium on demand, so it is slower than an HTML preview.
PDF_TIMEOUT = 180
PDF_MAX_BYTES = 10 * 1024 * 1024        # the backend's own ceiling, checked here for a better message


class BackendError(RuntimeError):
    """A request the backend refused, carrying a message worth showing the user."""


def _secret(name: str, default: str = "") -> str:
    """Streamlit secrets, then the environment. Secrets may be absent entirely."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass                                    # no secrets.toml — normal when running locally
    return os.environ.get(name, default).strip()


def base_url() -> str:
    return (st.session_state.get("api_base_url")
            or _secret("API_BASE_URL", "http://localhost")).rstrip("/")


def _headers() -> dict:
    key = st.session_state.get("api_key") or _secret("API_KEY")
    return {"X-API-Key": key} if key else {}


def _get(path: str, timeout: int = TIMEOUT, **params):
    try:
        r = requests.get(f"{base_url()}{path}", headers=_headers(),
                         params=params or None, timeout=timeout)
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {base_url()}: {e}") from e
    if r.status_code == 401:
        raise BackendError("The backend rejected the API key. Set API_KEY in secrets.")
    if not r.ok:
        raise BackendError(f"{r.status_code} from {path}: {r.text[:200]}")
    return r


def get_shelf() -> dict:
    """Base résumés with their tailored copies grouped underneath.

    The backend already assembles this for the React shelf, so the grouping the UI
    needs — which copy belongs to which original — is not re-derived here.
    """
    return _get("/api/resumes/shelf").json()


def get_preview_html(resume_id: str, template: str = "word") -> str:
    """One résumé rendered to HTML.

    `template` overrides what the résumé has stored for this render only. That is how
    every preview here comes back as Word Classic without writing to the résumé, so
    opening one in this app never changes how it looks in the React frontend.
    """
    return _get(f"/api/resumes/{resume_id}/preview",
                timeout=PREVIEW_TIMEOUT, template=template).text


def start_tailoring(base_resume_id: str, job_description: str) -> str:
    """Kick off a tailoring run against pasted JD text; returns its run id.

    The backend accepts either a job_id or raw job_description. This app only ever
    sends text, so nothing here needs a Job row to exist first.
    """
    try:
        r = requests.post(f"{base_url()}/api/resumes/tailor", headers=_headers(),
                          json={"base_resume_id": base_resume_id,
                                "job_description": job_description},
                          timeout=TIMEOUT)
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {base_url()}: {e}") from e
    if r.status_code == 409:
        raise BackendError("A tailoring run for this résumé is already going. Wait for it to finish.")
    if not r.ok:
        raise BackendError(f"{r.status_code}: {r.text[:300]}")
    run_id = r.json().get("run_id")
    if not run_id:
        raise BackendError(f"The backend accepted the request but returned no run id: {r.text[:200]}")
    return run_id


def get_run(run_id: str) -> dict:
    return _get(f"/api/monitor/run/{run_id}").json()


def import_pdf(filename: str, pdf_bytes: bytes) -> dict:
    """Upload a résumé PDF and get back the base résumé the backend structured from it.

    Unlike tailoring there is no run to poll — the endpoint does the extraction and the
    LLM call inline and returns the finished résumé, which is why this call alone can
    take the better part of a minute.
    """
    if not pdf_bytes:
        raise BackendError("That file is empty.")
    if len(pdf_bytes) > PDF_MAX_BYTES:
        raise BackendError(f"“{filename}” is {len(pdf_bytes) / 1e6:.1f} MB. The limit is 10 MB.")
    try:
        r = requests.post(f"{base_url()}/api/resumes/import-pdf", headers=_headers(),
                          files={"file": (filename, pdf_bytes, "application/pdf")},
                          timeout=IMPORT_TIMEOUT)
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {base_url()}: {e}") from e
    if not r.ok:
        detail = r.text[:300]
        try:
            detail = r.json().get("detail", detail)
        except Exception:
            pass
        raise BackendError(f"Import failed ({r.status_code}): {detail}")
    return r.json()


def delete_resume(resume_id: str) -> dict:
    """Delete one résumé, and say how many tailored copies went with it.

    Deleting a base cascade-deletes its tailored children on the backend, so the count
    in the reply is what the caller should report back to whoever pressed the button.
    """
    try:
        r = requests.delete(f"{base_url()}/api/resumes/{resume_id}",
                            headers=_headers(), timeout=TIMEOUT)
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {base_url()}: {e}") from e
    if r.status_code == 404:
        raise BackendError("That résumé is already gone.")
    if not r.ok:
        raise BackendError(f"Delete failed ({r.status_code}): {r.text[:200]}")
    try:
        return r.json()
    except Exception:
        return {"deleted": True, "id": resume_id, "children_deleted": 0}


def get_pdf(resume_id: str, template: str = "word") -> tuple:
    """One résumé as PDF bytes, with the filename the backend chose for it.

    The endpoint already composes a sensible download name ("Name_Resume.pdf") and sends
    it in Content-Disposition, so it is read back off the response rather than rebuilt
    here — that keeps a download from this app identical to one from the React frontend.
    """
    r = _get(f"/api/resumes/{resume_id}/pdf", timeout=PDF_TIMEOUT, template=template)
    name = ""
    disposition = r.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\'|")?([^";]+)"?', disposition)
    if match:
        name = unquote(match.group(1)).strip()
    return r.content, (name or f"{resume_id}.pdf")


# The model line-ups the toggle switches between. Each names both jobs explicitly
# rather than leaving one to fall through to the primary: an empty override follows
# `llm_provider`, which is not necessarily any of these.
ENGINES = {
    "OpenRouter": {
        "label": "OpenRouter — flash + gemma",
        "help": "deepseek-v4.1-flash structures, gemma-4-31b tailors. Fast; tailoring quality varies run to run.",
        "settings": {
            "parse_llm_provider": "openrouter",
            "parse_llm_model": "deepseek/deepseek-v4.1-flash",
            "cv_tailor_llm_provider": "openrouter",
            "cv_tailor_llm_model": "google/gemma-4-31b-it",
        },
    },
    "Codex": {
        "label": "Codex — gpt-5.6-luna",
        "help": "Both jobs on the Codex CLI with your ChatGPT plan. Slower, and local only — a hosted backend has no codex binary to run and no way to log one in.",
        "settings": {
            "parse_llm_provider": "codex_cli",
            "parse_llm_model": "gpt-5.6-luna",
            "cv_tailor_llm_provider": "codex_cli",
            "cv_tailor_llm_model": "gpt-5.6-luna",
        },
    },
    "Claude Code": {
        "label": "Claude Code — sonnet-5.5",
        "help": "Both jobs on the Claude Code CLI with your Claude plan, agent features switched off. About 30s per job, and local only — a hosted backend has no claude binary to run and no way to log one in.",
        "settings": {
            "parse_llm_provider": "claude_code",
            "parse_llm_model": "claude-sonnet-5-5",
            "cv_tailor_llm_provider": "claude_code",
            "cv_tailor_llm_model": "claude-sonnet-5-5",
        },
    },
}


def get_settings() -> dict:
    """The settings the app reads, so the toggle can show what is actually configured."""
    return _get("/api/settings").json()


def set_engine(name: str) -> dict:
    """Point both résumé jobs at one engine.

    Goes through PATCH /api/settings, the same endpoint and the same validation the React
    frontend uses, so the two frontends cannot drift apart — and so an unknown key or a
    bad value is refused here rather than written and discovered later.
    """
    engine = ENGINES[name]
    try:
        r = requests.patch(f"{base_url()}/api/settings", headers=_headers(),
                           json=engine["settings"], timeout=TIMEOUT)
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {base_url()}: {e}") from e
    if not r.ok:
        detail = r.text[:300]
        try:
            detail = r.json().get("detail", detail)
        except Exception:
            pass
        raise BackendError(f"Could not switch engine ({r.status_code}): {detail}")
    return r.json()


def current_engine(settings: dict | None = None) -> str:
    """Which line-up the backend is on, or "" when it matches neither."""
    settings = settings if settings is not None else get_settings()
    for name, engine in ENGINES.items():
        if all(str(settings.get(k, "")) == v for k, v in engine["settings"].items()):
            return name
    return ""
