"""An ATS score from one model call: the model classifies, the code scores.

How it works:
  1. The model reads the résumé and the job posting and answers a fixed set of questions.
     For each question it first lists its evidence: what the posting asks for that the résumé shows, and
     what it asks for that the résumé does not show. Then it picks one category from a fixed list.
     It returns only that, as strict JSON.
  2. Code turns each pick into a share and adds the parts by weight. The model never states a number.

This is a separate way of scoring from line_match.py, which reads the posting line by line with Jev.
It uses the same weights and the same final formula, so the two can be compared.
"""
import json
import logging

from backend.analyzer import line_match
from backend.analyzer.llm_client import call_llm

logger = logging.getLogger("jobnavigator.llm_ats_score")

PROVIDER = "openrouter"
MODEL = "deepseek/deepseek-v4.1-flash"
# Room for the evidence lists and the pick of all five questions.
MOST_REPLY_TOKENS = 2500
RESUME_MAX_CHARS = 40_000
POSTING_MAX_CHARS = 100_000

# What the model costs, in US dollars per million tokens.
INPUT_USD_PER_MILLION_TOKENS = 0.035
OUTPUT_USD_PER_MILLION_TOKENS = 0.60


# ── The questions, and the categories the model picks from ─────────────────────
#
# Each category has the share it stands for. A share of None means the posting does not ask for that
# part at all, so the part is left out of the score.

ROLE_FIT = "role_fit"
SKILLS = line_match.SKILLS
RESPONSIBILITIES = line_match.RESPONSIBILITIES
EXPERIENCE = line_match.EXPERIENCE_PART
QUALIFICATIONS = line_match.QUALIFICATIONS

QUESTIONS = {
    ROLE_FIT: {
        "asks": "Does the resume show the candidate doing the kind of work this job is mainly about?",
        "categories": {
            "different_work": ("The candidate's work is of a different kind", 0.0),
            "edge_of_the_work": ("Some of the job's work appears, but as a side part of a different kind of work", 0.35),
            "same_work_with_gaps": ("The candidate does this kind of work, with clear gaps", 0.75),
            "same_work": ("The candidate has done this kind of work as their main job", 1.0),
        },
    },
    SKILLS: {
        "asks": "How much of the skills, tools and technologies the posting asks for has the candidate actually used?",
        "categories": {
            "none": ("None of them", 0.0),
            "few": ("A small part of them", 0.25),
            "about_half": ("About half of them", 0.5),
            "most": ("Most of them", 0.75),
            "all": ("All of them, as work the candidate actually did", 1.0),
        },
    },
    RESPONSIBILITIES: {
        "asks": "How much of the duties the posting describes has the candidate actually done?",
        "categories": {
            "none": ("None of them", 0.0),
            "few": ("A small part of them", 0.25),
            "about_half": ("About half of them", 0.5),
            "most": ("Most of them", 0.75),
            "all": ("All of them", 1.0),
            "not_stated": ("The posting describes no duties", None),
        },
    },
    EXPERIENCE: {
        "asks": "Does the candidate have the years and level the posting asks for, counting only years spent "
                "doing this kind of work?",
        "categories": {
            "far_below": ("Far below what is asked", 0.0),
            "below": ("Somewhat below what is asked", 0.5),
            "meets": ("Meets or exceeds what is asked", 1.0),
            "not_stated": ("The posting states no years or level", None),
        },
    },
    QUALIFICATIONS: {
        "asks": "Does the candidate have the degrees and certifications the posting asks for?",
        "categories": {
            "none": ("None of them", 0.0),
            "some": ("Some of them", 0.5),
            "all": ("All of them", 1.0),
            "not_stated": ("The posting asks for no degree or certification", None),
        },
    },
}

SYSTEM_PROMPT = (
    "You compare a resume with a job posting. For each question, first give your evidence, then pick exactly "
    "one of its categories.\n"
    "- matched: each thing the posting asks for that the resume shows, written as the posting's item followed "
    "by the resume's own words that show it.\n"
    "- missing: each thing the posting asks for that the resume does not show.\n"
    "- reason: one sentence saying why the evidence leads to the category you pick.\n"
    "Use only what is written in the two texts. Judge the work the candidate did, not the words used: the same "
    "work described differently counts, and a skill that is only listed with no work behind it counts for less. "
    "Reply with the JSON object only."
)

# What the model gives for each question, in the order it writes them: the evidence comes before the pick.
MATCHED = "matched"
MISSING = "missing"
REASON = "reason"
CATEGORY = "category"


def _schema_of_the_reply() -> dict:
    """Strict JSON: for every question, the evidence lists, a reason, and one of that question's categories."""
    properties = {}
    for question, details in QUESTIONS.items():
        properties[question] = {
            "type": "object",
            "additionalProperties": False,
            "required": [MATCHED, MISSING, REASON, CATEGORY],
            "properties": {
                MATCHED: {"type": "array", "items": {"type": "string"}},
                MISSING: {"type": "array", "items": {"type": "string"}},
                REASON: {"type": "string"},
                CATEGORY: {"type": "string", "enum": list(details["categories"])},
            },
        }
    return {"type": "object", "additionalProperties": False, "required": list(QUESTIONS), "properties": properties}


def _prompt(resume_text: str, job_text: str) -> str:
    question_blocks = []
    for question, details in QUESTIONS.items():
        category_lines = [f"  - {category}: {meaning}" for category, (meaning, _share) in details["categories"].items()]
        question_blocks.append(f"{question}: {details['asks']}\n" + "\n".join(category_lines))
    posting = line_match.posting_as_plain_text(job_text)[:POSTING_MAX_CHARS]
    resume = (resume_text or "")[:RESUME_MAX_CHARS]
    return ("QUESTIONS\n\n" + "\n\n".join(question_blocks)
            + f"\n\nJOB POSTING\n<<<\n{posting}\n>>>\n\nRESUME\n<<<\n{resume}\n>>>")


# ── Asking the model ───────────────────────────────────────────────────────────

async def ats_score(resume_text: str, job_text: str, model: str = MODEL) -> dict | None:
    """The score out of 100, the model's pick and evidence for each question, and what the call cost.

    Returns {"score", "role_fit", "total_of_parts", "parts", "picks", "evidence", "input_tokens",
    "output_tokens", "cost_usd"}, where evidence is {question: {"matched": [...], "missing": [...], "reason"}}.
    Returns None when the model cannot be asked or its reply is not usable.
    """
    try:
        reply = await call_llm(_prompt(resume_text, job_text), SYSTEM_PROMPT, max_tokens=MOST_REPLY_TOKENS,
                               provider=PROVIDER, model=model, response_schema=_schema_of_the_reply(),
                               schema_name="ats_classification", temperature=0, reasoning=False)
    except Exception as error:
        logger.warning(f"The ATS classification call failed ({type(error).__name__}: {error})")
        return None

    answers = _answers_in(reply.get("text") or "")
    if answers is None:
        logger.warning(f"The ATS classification reply was not usable: {(reply.get('text') or '')[:200]!r}")
        return None

    picks = {question: answer[CATEGORY] for question, answer in answers.items()}
    evidence = {}
    for question, answer in answers.items():
        evidence[question] = {MATCHED: list(answer.get(MATCHED) or []), MISSING: list(answer.get(MISSING) or []),
                              REASON: str(answer.get(REASON) or "")}

    usage = reply.get("usage") or {}
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    return {**score_from_picks(picks), "picks": picks, "evidence": evidence, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "cost_usd": cost_in_usd(input_tokens, output_tokens)}


def _answers_in(reply_text: str) -> dict | None:
    """The model's answer to every question, or None when one is missing or its category is not a known one."""
    try:
        answers = json.loads(reply_text[reply_text.index("{"):reply_text.rindex("}") + 1])
    except ValueError:
        return None
    for question, details in QUESTIONS.items():
        answer = answers.get(question)
        if not isinstance(answer, dict) or answer.get(CATEGORY) not in details["categories"]:
            return None
    return {question: answers[question] for question in QUESTIONS}


# ── Scoring, in code ───────────────────────────────────────────────────────────

def score_from_picks(picks: dict) -> dict:
    """The score from the model's picks: the parts added by weight, then scaled by role fit."""
    shares = {question: QUESTIONS[question]["categories"][pick][1] for question, pick in picks.items()}
    role_fit = shares.pop(ROLE_FIT)
    total_of_parts = line_match._parts_added_by_weight(shares)
    return {"score": line_match._score_out_of_100(role_fit, total_of_parts), "role_fit": role_fit,
            "total_of_parts": total_of_parts, "parts": shares}


def cost_in_usd(input_tokens: int, output_tokens: int) -> float:
    input_cost = input_tokens * INPUT_USD_PER_MILLION_TOKENS / 1_000_000
    output_cost = output_tokens * OUTPUT_USD_PER_MILLION_TOKENS / 1_000_000
    return round(input_cost + output_cost, 6)
