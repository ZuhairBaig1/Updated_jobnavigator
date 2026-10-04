"""ATS fit score from a decision model (Jev or GLiDE). The model answers five questions; this code turns the answers into a 0-100 score."""
import logging
import os

import httpx

from backend.analyzer.prompt_fence import fence
from backend.models.db import SessionLocal, Setting

logger = logging.getLogger("jobnavigator.decision_scorer")

SETTING_KEY = "ats_scorer"
LLM_SCORER = "llm"
DEFAULT_SCORER = "jev"

SCORERS = {
    "jev": {
        "label": "Jev",
        "url": "https://openrouter.ai/api/alpha/decisions",
        "model": "~typesafe/jev-latest",
        "api_key_env_var": "OPENROUTER_API_KEY",
        "provider_for_call_log": "openrouter",
    },
    "glide": {
        "label": "GLiDE",
        "url": "https://api.fastino.ai/v1/systemone",
        "model": "fastino/glide",
        "api_key_env_var": "FASTINO_API_KEY",
        "provider_for_call_log": "fastino",
    },
}

# Names the app shows in the score breakdown, one per question.
BREAKDOWN_LABELS = {
    "keyword_coverage": "keywords",
    "domain": "domain",
    "impact_evidence": "impact",
    "experience": "experience",
    "requirements": "requirements",
}

REQUEST_TIMEOUT_SECONDS = 120
POINTS_PER_QUESTION = 20

RESUME_MAX_CHARS = 40_000
JOB_MAX_CHARS = 8_000

OPENROUTER_KEY_PREFIX = "sk-or-"
SETTINGS_THAT_MAY_HOLD_AN_OPENROUTER_KEY = (
    "llm_api_key",
    "scoring_llm_api_key",
    "cv_tailor_llm_api_key",
    "parse_llm_api_key",
)

# Graded questions: the model picks one level, listed worst to best.
GRADED_QUESTIONS = {
    "keyword_coverage": (
        "How well does the resume use the job description's named tools, technologies, "
        "and required skills?",
        [
            "Almost none are present, including similar skills",
            "A few are present; most required skills are missing",
            "Most are present, using similar or exact wording",
            "Nearly all are present using exact wording",
        ],
    ),
    "domain": (
        "How strongly does the candidate's experience match the job's industry or domain?",
        [
            "No evidence of relevant industry or domain experience",
            "Only adjacent or tangential experience",
            "Some direct experience, but it is not the primary focus",
            "Strong, direct, sustained experience in the exact domain",
        ],
    ),
    "impact_evidence": (
        "How concrete and quantified are the resume bullets relevant to this job?",
        [
            "No bullets relevant to this job",
            "Relevant bullets are generic duties, with no specifics and no numbers",
            "Relevant bullets are specific and concrete (named systems, scale, scope), "
            "but have few or no numbers",
            "Relevant bullets show clear, quantified results",
        ],
    ),
}

# True/false questions: the model answers with a probability from 0 to 1.
TRUE_FALSE_QUESTIONS = {
    "experience": (
        "How well do the candidate's years of experience and seniority match the job's "
        "stated requirements?",
        {
            "true": "The candidate's experience level and seniority reasonably match the job",
            "false": "The candidate is clearly below or above the job's stated experience "
                     "or seniority level",
        },
    ),
    "requirements": (
        "Does the candidate meet the job's stated mandatory requirements, such as education, "
        "certifications, or clearance?",
        {
            "true": "The resume provides evidence that the candidate meets the stated "
                    "mandatory requirements",
            "false": "The resume provides no evidence of meeting, or clearly fails, one or "
                     "more stated mandatory requirements",
        },
    ),
}

ALL_QUESTION_KEYS = list(GRADED_QUESTIONS) + list(TRUE_FALSE_QUESTIONS)


def ats_scorer(db=None) -> str:
    """Who produces the ATS score: "llm", or a key of SCORERS. Jev when the setting is missing."""
    owns_session = db is None
    if owns_session:
        db = SessionLocal()
    try:
        row = db.query(Setting).filter(Setting.key == SETTING_KEY).first()
        value = row.value if row and row.value else DEFAULT_SCORER
        scorer = value.strip().lower()
        if scorer in SCORERS:
            return scorer
        return LLM_SCORER
    finally:
        if owns_session:
            db.close()


def _find_api_key(scorer: str) -> str:
    env_var = SCORERS[scorer]["api_key_env_var"]
    key = os.environ.get(env_var, "").strip()
    if key:
        return key
    if scorer == "jev":
        return _find_stored_openrouter_key()
    return ""


def _find_stored_openrouter_key() -> str:
    """An OpenRouter key already saved for one of the LLM features, or ""."""
    db = SessionLocal()
    try:
        for setting_name in SETTINGS_THAT_MAY_HOLD_AN_OPENROUTER_KEY:
            row = db.query(Setting).filter(Setting.key == setting_name).first()
            value = (row.value if row and row.value else "").strip()
            if value.startswith(OPENROUTER_KEY_PREFIX):
                return value
        return ""
    finally:
        db.close()


def _build_state(resume_text: str, job_text: str) -> str:
    resume_block = fence(resume_text[:RESUME_MAX_CHARS], "CANDIDATE RESUME")
    job_block = fence(job_text[:JOB_MAX_CHARS], "JOB POSTING")
    return f"{resume_block}\n\n{job_block}"


def _build_questions() -> dict:
    questions = {}
    for key, (instructions, levels) in GRADED_QUESTIONS.items():
        questions[key] = {"type": "score", "instructions": instructions, "criteria": levels}
    for key, (instructions, criteria) in TRUE_FALSE_QUESTIONS.items():
        questions[key] = {"type": "noul", "instructions": instructions, "criteria": criteria}
    return questions


async def _ask_model(scorer: str, api_key: str, state: str) -> dict:
    config = SCORERS[scorer]
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(
            config["url"],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": config["model"], "state": state, "questions": _build_questions()},
        )
    response.raise_for_status()
    return response.json()


def _graded_answer_to_fraction(answer: dict, level_count: int) -> float:

    level = float(answer.get("expected_level", answer["score"]))
    highest_level = level_count - 1
    return _clamp_to_unit_range(level / highest_level)


def _true_false_answer_to_fraction(answer: dict) -> float:
    return _clamp_to_unit_range(float(answer["noul"]))


def _clamp_to_unit_range(value: float) -> float:
    return max(0.0, min(1.0, value))


async def score_resume(scorer: str, resume_text: str, job_text: str) -> dict | None:
    """Returns {"score", "breakdown", "answers", "confidence", "usage"}, or None on any failure."""
    label = SCORERS[scorer]["label"]

    api_key = _find_api_key(scorer)
    if not api_key:
        env_var = SCORERS[scorer]["api_key_env_var"]
        logger.warning(f"{label} scoring is selected but no API key was found ({env_var}) — skipping")
        return None

    try:
        reply = await _ask_model(scorer, api_key, _build_state(resume_text, job_text))
    except Exception as error:
        logger.warning(f"{label} scoring call failed: {type(error).__name__}: {error}")
        return None

    answers = reply.get("answers") or {}
    usage = reply.get("usage") or {}

    if not isinstance(answers, dict) or not all(key in answers for key in ALL_QUESTION_KEYS):
        logger.warning(f"{label} scoring returned an incomplete reply: {answers!r}")
        return None

    answer_fractions = {}
    confidence = {}
    try:
        for key, (_, levels) in GRADED_QUESTIONS.items():
            answer_fractions[key] = _graded_answer_to_fraction(answers[key], len(levels))
            confidence[key] = answers[key].get("confidence")
        for key in TRUE_FALSE_QUESTIONS:
            answer_fractions[key] = _true_false_answer_to_fraction(answers[key])
            confidence[key] = answers[key].get("confidence")
    except (TypeError, ValueError, KeyError) as error:
        logger.warning(f"{label} scoring: unreadable answer ({error}): {answers!r}")
        return None

    points = {key: fraction * POINTS_PER_QUESTION for key, fraction in answer_fractions.items()}
    breakdown = {BREAKDOWN_LABELS[key]: round(value) for key, value in points.items()}
    return {
        "score": round(sum(points.values())),
        "breakdown": breakdown,
        "answers": answer_fractions,
        "confidence": confidence,
        "usage": usage,
    }
