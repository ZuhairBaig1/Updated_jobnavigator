"""ATS fit score from a decision model (Jev or GLiDE). The model answers a set of questions; this code turns the answers into a 0-100 score."""
import asyncio
import logging
import os

import httpx

from backend.analyzer.job_requirements import describe_requirements
from backend.analyzer.prompt_fence import fence
from backend.models.db import SessionLocal, Setting

logger = logging.getLogger("jobnavigator.decision_scorer")

SETTING_KEY = "ats_scorer"
LLM_SCORER = "llm"
DEFAULT_SCORER = "jev"

# What the model reads about the job: the raw posting text, its extracted requirements,
# or, for each question, only the sections of the posting that question is about.
JOB_CONTEXT_SETTING_KEY = "ats_scorer_job_context"
EXTRACTED_REQUIREMENTS = "requirements"
RAW_POSTING = "posting"
POSTING_SECTIONS = "sections"

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
    "core_match": "main work",
    "skill_match": "skills",
    "responsibility_match": "responsibilities",
    "experience_match": "experience",
    "requirements_match": "requirements",
    "domain_match": "domain",
}

REQUEST_TIMEOUT_SECONDS = 120
# The breakdown shows every question on the same 0-20 scale, whatever its weight.
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
    "core_match": (
        "Going by the job posting's title and the work it describes, what kind of work and which skills is "
        "this job mainly about? Does the resume show the candidate doing that kind of work with those skills?",
        [
            "No: the candidate's work is of a different kind and the job's main skills are absent",
            "Only at the edges: some of the job's main skills appear, but as a side part of a different kind of work",
            "Yes: the candidate has done real work of this kind with the job's main skills",
            "Yes: this kind of work with these skills is what the candidate mainly does",
        ],
    ),

    "skill_match": (
        "How much of the job's important technical skill requirements are "
        "demonstrated by the resume? Consider exact matches, reasonable "
        "equivalents, and clearly transferable technologies. Prioritize core "
        "and important skills over minor or preferred keywords.",
        [
            "Very little of the important technical skills are demonstrated",
            "Less than half of the important technical skills are demonstrated",
            "About half to most of the important technical skills are demonstrated",
            "Most or nearly all important/core technical skills are demonstrated",
        ],
    ),

    "responsibility_match": (
        "How much of the job's core responsibilities are demonstrated by the "
        "candidate's previous work or project experience? Evaluate actual work "
        "performed, including closely related or transferable responsibilities.",
        [
            "Very little overlap with the job's core responsibilities",
            "Less than half of the core responsibilities are demonstrated",
            "About half to most of the core responsibilities are demonstrated",
            "Most or nearly all major responsibilities are demonstrated",
        ],
    ),

    "experience_match": (
        "How well does the candidate's overall experience match the level, type, "
        "and scope of experience expected for this job? Consider years of "
        "experience, seniority, technical depth, and relevance of previous work. "
        "Count only the years the candidate spent doing the kind of work this job "
        "requires; years in an unrelated field do not count towards the years asked "
        "for. Do not require an exact job-title match.",
        [
            "The candidate is substantially below the required level or their experience is largely unrelated",
            "The candidate has some relevant experience but is noticeably below or different from the target",
            "The candidate's experience is generally appropriate for the role, with some gaps",
            "The candidate's experience level, technical depth, and type of work are a strong match",
        ],
    ),

    "requirements_match": (
        "How well does the resume satisfy the job's important stated requirements? "
        "Prioritize explicitly mandatory and core requirements over preferred, "
        "minor, or supplementary qualifications. Consider reasonable equivalents "
        "and transferable experience. Do not penalize the candidate simply because "
        "a long posting contains many optional requirements.",
        [
            "Very few of the important/core requirements are satisfied",
            "Less than half of the important/core requirements are satisfied",
            "About half to most of the important/core requirements are satisfied",
            "Most or nearly all important/core requirements are satisfied; remaining gaps are mainly minor or preferred",
        ],
    ),

    "domain_match": (
        "How relevant is the candidate's previous industry, business, or technical "
        "domain experience to this job? Give credit for closely related or "
        "transferable domain experience. An exact industry match is not required "
        "unless explicitly required by the job.",
        [
            "Little or no relevant domain experience",
            "Mostly adjacent or transferable domain experience",
            "Meaningful direct or closely related domain experience",
            "Strong and directly relevant domain experience",
        ],
    ),
}


TRUE_FALSE_QUESTIONS = {
    "experience": (
        "Is the candidate's experience in the kind of work this job requires "
        "reasonably appropriate for the seniority and years-of-experience "
        "expectations of this job? Years in an unrelated field do not count.",
        {
            "true": (
                "The candidate is reasonably within the expected experience "
                "and seniority range"
            ),
            "false": (
                "The candidate is clearly outside the expected experience "
                "or seniority range"
            ),
        },
    ),

    "requirements": (
        "Does the resume provide sufficient evidence that the candidate meets "
        "the job's explicitly mandatory qualifications?",
        {
            "true": (
                "The resume provides evidence supporting the stated mandatory "
                "qualifications"
            ),
            "false": (
                "The resume clearly fails or provides insufficient evidence for "
                "one or more explicitly mandatory qualifications"
            ),
        },
    ),
}


# The share of the 0-100 score each question carries. A question without a weight is still
# asked and reported, but does not move the score.
QUESTION_WEIGHTS = {
    "core_match": 0.25,
    "skill_match": 0.20,
    "responsibility_match": 0.20,
    "experience_match": 0.15,
    "requirements_match": 0.15,
    "domain_match": 0.05,
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


def _job_context_setting(db=None) -> str:
    owns_session = db is None
    if owns_session:
        db = SessionLocal()
    try:
        row = db.query(Setting).filter(Setting.key == JOB_CONTEXT_SETTING_KEY).first()
        value = row.value if row and row.value else RAW_POSTING
        return value.strip().lower()
    finally:
        if owns_session:
            db.close()


def uses_extracted_requirements(db=None) -> bool:
    """Whether the model reads the posting's extracted requirements instead of its raw text."""
    return _job_context_setting(db) == EXTRACTED_REQUIREMENTS


def uses_posting_sections(db=None) -> bool:
    """Whether each question is asked with only the sections of the posting it is about."""
    return _job_context_setting(db) == POSTING_SECTIONS


def _build_state(resume_text: str, job_text: str, requirements: dict | None = None) -> str:
    resume_block = fence(resume_text[:RESUME_MAX_CHARS], "CANDIDATE RESUME")
    if requirements:
        job_block = fence(describe_requirements(requirements), "JOB REQUIREMENTS")
    else:
        job_block = fence(job_text[:JOB_MAX_CHARS], "JOB POSTING")
    return f"{resume_block}\n\n{job_block}"


def _build_questions() -> dict:
    questions = {}
    for key, (instructions, levels) in GRADED_QUESTIONS.items():
        questions[key] = {"type": "score", "instructions": instructions, "criteria": levels}
    for key, (instructions, criteria) in TRUE_FALSE_QUESTIONS.items():
        questions[key] = {"type": "noul", "instructions": instructions, "criteria": criteria}
    return questions


async def _ask_model(scorer: str, api_key: str, state: str, questions: dict | None = None) -> dict:
    config = SCORERS[scorer]
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.post(
            config["url"],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": config["model"], "state": state, "questions": questions or _build_questions()},
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


async def score_resume(scorer: str, resume_text: str, job_text: str,
                       requirements: dict | None = None) -> dict | None:
    """Returns {"score", "breakdown", "answers", "confidence", "usage"}, or None on any failure.

    The model reads the posting's extracted requirements when given, else the raw posting text.
    """
    label = SCORERS[scorer]["label"]

    api_key = _find_api_key(scorer)
    if not api_key:
        env_var = SCORERS[scorer]["api_key_env_var"]
        logger.warning(f"{label} scoring is selected but no API key was found ({env_var}) — skipping")
        return None

    try:
        reply = await _ask_model(scorer, api_key, _build_state(resume_text, job_text, requirements))
    except Exception as error:
        logger.warning(f"{label} scoring call failed: {type(error).__name__}: {error}")
        return None

    return _score_from_answers(label, reply.get("answers") or {}, reply.get("usage") or {})


def _score_from_answers(label: str, answers: dict, usage: dict) -> dict | None:
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

    weighted_total = sum(answer_fractions[key] * weight for key, weight in QUESTION_WEIGHTS.items())
    breakdown = {label: round(answer_fractions[key] * POINTS_PER_QUESTION)
                 for key, label in BREAKDOWN_LABELS.items()}
    return {
        "score": round(weighted_total * 100),
        "breakdown": breakdown,
        "answers": answer_fractions,
        "confidence": confidence,
        "usage": usage,
    }


# ── Asking each question with only the part of the posting it is about ─────────

# The sections each question reads, as (label shown to the model, key in the split posting).
SECTIONS_BY_QUESTION = {
    "core_match": [("About the role", "general"),
                   ("Required skills and qualifications", "must_have"),
                   ("Responsibilities", "responsibilities")],
    "skill_match": [("Required skills and qualifications", "must_have"),
                    ("Preferred, not required", "nice_to_have")],
    "responsibility_match": [("Responsibilities", "responsibilities")],
    "experience_match": [("What the posting says about experience level", "experience_level"),
                         ("The kind of experience required", "must_have")],
    "requirements_match": [("Required skills and qualifications", "must_have"),
                           ("Education", "education"),
                           ("Certifications", "certifications"),
                           ("Work authorization", "work_authorization")],
    "domain_match": [("About the company", "about"),
                     ("About the role", "general"),
                     ("Responsibilities", "responsibilities")],
    "experience": [("What the posting says about experience level", "experience_level"),
                   ("The kind of experience required", "must_have")],
    "requirements": [("Required skills and qualifications", "must_have"),
                     ("Education", "education"),
                     ("Certifications", "certifications"),
                     ("Work authorization", "work_authorization")],
}

def job_context_for_question(question_key: str, sections: dict, job_text: str) -> str:
    """The part of the posting one question is about; the whole posting only when it has no such part."""
    blocks = []
    for label, section_key in SECTIONS_BY_QUESTION[question_key]:
        lines = sections.get(section_key) or []
        if lines:
            blocks.append(f"{label}:\n" + "\n".join(f"- {line}" for line in lines))
    if question_key in ("experience_match", "experience") and sections.get("years"):
        blocks.insert(0, f"Years of experience asked for: {sections['years']}")
    return "\n\n".join(blocks) or job_text


async def _ask_one_question(scorer: str, api_key: str, question_key: str, resume_text: str, job_context: str) -> dict:
    state = (fence(resume_text[:RESUME_MAX_CHARS], "CANDIDATE RESUME") + "\n\n"
             + fence(job_context[:JOB_MAX_CHARS], "JOB POSTING"))
    return await _ask_model(scorer, api_key, state, {question_key: _build_questions()[question_key]})


async def score_resume_by_posting_section(scorer: str, resume_text: str, job_text: str,
                                          sections: dict | None = None) -> dict | None:
    """Like score_resume, but each question is asked alone, reading only its own part of the posting."""
    from backend.analyzer.job_sections import split_posting

    label = SCORERS[scorer]["label"]
    api_key = _find_api_key(scorer)
    if not api_key:
        logger.warning(f"{label} scoring is selected but no API key was found — skipping")
        return None

    sections = sections or split_posting(job_text)
    try:
        replies = await asyncio.gather(*[
            _ask_one_question(scorer, api_key, key, resume_text, job_context_for_question(key, sections, job_text))
            for key in ALL_QUESTION_KEYS])
    except Exception as error:
        logger.warning(f"{label} scoring call failed: {type(error).__name__}: {error}")
        return None

    answers, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    for reply in replies:
        answers.update(reply.get("answers") or {})
        for field in usage:
            usage[field] += (reply.get("usage") or {}).get(field, 0)
    return _score_from_answers(label, answers, usage)


# ── Question set 2: one graded question per requirement the posting states ─────
#
# Each level is (what the model can pick, the share of that item's points it earns).

QUESTION_SET_SETTING_KEY = "ats_scorer_question_set"
FIVE_QUESTIONS = "five_questions"
PER_REQUIREMENT = "per_requirement"

SKILL_LEVELS = [
    ("Not mentioned anywhere in the resume", 0.0),
    ("Only listed in a skills section or summary, never shown in use", 0.4),
    ("Used in at least one role or project", 0.8),
    ("Used across several roles, or central to the candidate's work", 1.0),
]

DUTY_LEVELS = [
    ("No evidence of this kind of work", 0.0),
    ("Related or adjacent work only", 0.4),
    ("Has done this", 0.8),
    ("Has done this repeatedly or led it", 1.0),
]

CERTIFICATION_LEVELS = [
    ("Not held", 0.0),
    ("Holds a related or lower-level certification in the same area", 0.4),
    ("Holds this certification", 1.0),
]

YEARS_LEVELS = [
    ("Less than half the years asked for", 0.0),
    ("Somewhat fewer years than asked for", 0.5),
    ("About the years asked for", 1.0),
    ("Clearly more years than asked for", 1.0),
]

EDUCATION_LEVELS = [
    ("No education shown, or it does not meet what is asked", 0.0),
    ("A lower level or an unrelated field", 0.5),
    ("Meets or exceeds what is asked", 1.0),
]

WORK_AUTHORIZATION_LEVELS = [
    ("The resume shows the candidate lacks it or needs sponsorship", 0.0),
    ("The resume does not say", 0.5),
    ("The resume states a matching authorization", 1.0),
]

SECTION_WEIGHTS = {
    "core_must_have": 30,
    "must_have": 15,
    "responsibilities": 15,
    "years": 10,
    "nice_to_have": 10,
    "required_certifications": 8,
    "education": 5,
    "work_authorization": 4,
    "preferred_certifications": 3,
}

# The two must-have groups together; a posting with only one of them gives it all of these points.
MUST_HAVE_GROUPS = ("core_must_have", "must_have")

SECTION_LABELS = {
    "core_must_have": "core must-haves",
    "must_have": "other must-haves",
    "responsibilities": "responsibilities",
    "years": "years",
    "nice_to_have": "nice-to-haves",
    "required_certifications": "required certs",
    "education": "education",
    "work_authorization": "work authorization",
    "preferred_certifications": "preferred certs",
}

MAX_QUESTIONS_PER_REQUEST = 50


def uses_per_requirement_questions(db=None) -> bool:
    """Whether scoring asks one question per requirement (set 2) instead of the five questions."""
    owns_session = db is None
    if owns_session:
        db = SessionLocal()
    try:
        row = db.query(Setting).filter(Setting.key == QUESTION_SET_SETTING_KEY).first()
        value = row.value if row and row.value else FIVE_QUESTIONS
        return value.strip().lower() == PER_REQUIREMENT
    finally:
        if owns_session:
            db.close()


def _named_with_aliases(item: dict) -> str:
    other_names = ", ".join(item.get("aliases", [])[:2])
    return f"{item['name']} (also called: {other_names})" if other_names else item["name"]


def _skill_question(item: dict) -> str:
    return f"How much evidence does the resume show of this skill or qualification: {_named_with_aliases(item)}?"


def build_requirement_questions(requirements: dict) -> list:
    """One question per must-have, nice-to-have, certification and responsibility, plus years, education and work authorization when the posting states them."""
    questions = []

    def add(section, item_label, instructions, levels):
        questions.append({"key": f"q{len(questions)}", "section": section, "item": item_label,
                          "instructions": instructions, "levels": levels})

    for item in requirements.get("must_have") or []:
        add("core_must_have" if item.get("core") is True else "must_have", item["name"],
            _skill_question(item), SKILL_LEVELS)

    for item in requirements.get("nice_to_have") or []:
        add("nice_to_have", item["name"], _skill_question(item), SKILL_LEVELS)

    for certification in requirements.get("certifications") or []:
        section = "required_certifications" if certification["required"] else "preferred_certifications"
        add(section, certification["name"],
            f"Does the candidate hold this certification: {certification['name']}?", CERTIFICATION_LEVELS)

    for duty in requirements.get("responsibilities") or []:
        add("responsibilities", duty,
            f"How much evidence does the resume show that the candidate has done this kind of work: {duty}?",
            DUTY_LEVELS)

    if requirements.get("min_years"):
        years = requirements["min_years"]
        add("years", f"{years}+ years",
            f"The job asks for at least {years} years of relevant experience. How does the candidate's "
            f"relevant experience compare?", YEARS_LEVELS)

    if requirements.get("education"):
        add("education", requirements["education"],
            f"The job asks for this education: {requirements['education']}. How does the candidate's "
            f"education compare?", EDUCATION_LEVELS)

    if requirements.get("work_authorization"):
        add("work_authorization", requirements["work_authorization"],
            f"The job asks for this work authorization: {requirements['work_authorization']}. "
            f"What does the resume show?", WORK_AUTHORIZATION_LEVELS)

    return questions


def _level_probabilities(answer: dict, level_count: int) -> list | None:
    """How likely the model thinks each level is, in level order; None when it did not say."""
    probabilities = answer.get("probabilities")
    if isinstance(probabilities, dict):
        probabilities = [probabilities.get(str(level), 0) for level in range(level_count)]
    if not isinstance(probabilities, list) or len(probabilities) != level_count or sum(probabilities) <= 0:
        return None
    return [float(probability) / sum(probabilities) for probability in probabilities]


def _share_earned(answer: dict, levels: list) -> float:
    """The share of an item's points earned, weighting each level by how likely the model thinks it is."""
    shares = [share for _, share in levels]
    probabilities = _level_probabilities(answer, len(levels))
    if probabilities:
        return sum(probability * share for probability, share in zip(probabilities, shares))
    position = max(0.0, min(float(len(shares) - 1), float(answer["score"])))
    lower = int(position)
    upper = min(lower + 1, len(shares) - 1)
    return shares[lower] + (shares[upper] - shares[lower]) * (position - lower)


def _most_likely_level(answer: dict, levels: list) -> str:
    probabilities = _level_probabilities(answer, len(levels))
    index = probabilities.index(max(probabilities)) if probabilities else round(float(answer["score"]))
    return levels[max(0, min(len(levels) - 1, index))][0]


async def _ask_in_batches(scorer: str, api_key: str, state: str, questions: list) -> tuple:
    batches = [questions[start:start + MAX_QUESTIONS_PER_REQUEST]
               for start in range(0, len(questions), MAX_QUESTIONS_PER_REQUEST)]
    replies = await asyncio.gather(*[
        _ask_model(scorer, api_key, state, {
            question["key"]: {"type": "score", "instructions": question["instructions"],
                              "criteria": [label for label, _ in question["levels"]]}
            for question in batch})
        for batch in batches])
    answers, usage = {}, {"input_tokens": 0, "output_tokens": 0}
    for reply in replies:
        answers.update(reply.get("answers") or {})
        for field in usage:
            usage[field] += (reply.get("usage") or {}).get(field, 0)
    return answers, usage


async def score_resume_by_requirement(scorer: str, resume_text: str, requirements: dict) -> dict | None:
    """Question set 2. Returns {"score", "breakdown", "sections", "items", "usage"}, or None on any failure."""
    label = SCORERS[scorer]["label"]
    questions = build_requirement_questions(requirements)
    api_key = _find_api_key(scorer)
    if not questions or not api_key:
        logger.warning(f"{label} per-requirement scoring skipped: "
                       f"{'no API key' if questions else 'the posting has no requirements to ask about'}")
        return None

    state = fence(resume_text[:RESUME_MAX_CHARS], "CANDIDATE RESUME")
    try:
        answers, usage = await _ask_in_batches(scorer, api_key, state, questions)
        items = [{"section": question["section"], "item": question["item"],
                  "level": _most_likely_level(answers[question["key"]], question["levels"]),
                  "share": round(_share_earned(answers[question["key"]], question["levels"]), 2),
                  "confidence": answers[question["key"]].get("confidence")}
                 for question in questions]
    except Exception as error:
        logger.warning(f"{label} per-requirement scoring failed: {type(error).__name__}: {error}")
        return None

    sections = _points_by_section(items)
    return {
        "score": round(sum(section["points"] for section in sections.values())),
        "breakdown": {SECTION_LABELS[name]: round(section["share"] * POINTS_PER_QUESTION)
                      for name, section in sections.items()},
        "sections": sections,
        "items": items,
        "answers": {},
        "confidence": {},
        "usage": usage,
    }


def _points_by_section(items: list) -> dict:
    """Each section's points, with the sections the posting has scaled up to fill 100."""
    sections = {}
    for section, weight in SECTION_WEIGHTS.items():
        shares = [item["share"] for item in items if item["section"] == section]
        if shares:
            sections[section] = {"weight": weight, "share": sum(shares) / len(shares), "items": len(shares)}
    _give_a_lone_must_have_group_all_must_have_points(sections)
    total_weight = sum(section["weight"] for section in sections.values())
    for section in sections.values():
        section["points"] = round(100 * section["weight"] * section["share"] / total_weight, 1)
        section["out_of"] = round(100 * section["weight"] / total_weight, 1)
    return sections


def _give_a_lone_must_have_group_all_must_have_points(sections: dict) -> None:
    present = [group for group in MUST_HAVE_GROUPS if group in sections]
    if len(present) == 1:
        sections[present[0]]["weight"] = sum(SECTION_WEIGHTS[group] for group in MUST_HAVE_GROUPS)
