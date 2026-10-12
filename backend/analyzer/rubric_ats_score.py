"""An ATS score from one small model call and a keyword check: the model picks, the code scores.

How it works:
  1. Code checks the posting's keywords against the résumé, word for word: used in a role, only listed
     as a skill, or not there at all. That is the keyword match, and it costs nothing.
  2. One model call answers a short rubric. For each question the model gives one short reason and then
     picks one option. That is the contextual match and the title match. It never states a number.
  3. The model also names the few skills the job is mainly about. Code looks for each one in the roles,
     word for word, and the model's pick for core skills can be no higher than the share it finds: asked
     alone, the model gave a résumé that only lists Kubernetes the same pick as one that ran it.
  4. Code turns each pick into a share, adds the parts by weight, and scales the total by whether the
     candidate's career is in this occupation and by how many of the job's core skills they have used.

What keeps it steady:
  - The model is asked several times at once. Each question earns the average of what its picks stand for,
    so a question the model is in two minds about moves the score a little, not a whole step; the middle
    pick is the one shown. Asked once, the same résumé scored 82 one time and 92 the next.
  - What code can settle, code settles: the keywords, the core skills in the roles, and the years of work,
    which are added up from the roles' dates and set against the years the posting asks for.
  - A résumé and posting scored once are remembered, so asking again gives the same score.

This is a third way of scoring. line_match.py asks Jev about every line of the posting, which is steadier
but takes a dozen calls; this one takes a single call, so it is quick and cheap.
"""
import asyncio
import hashlib
import json
import logging
import re
from datetime import date

from backend.analyzer import keyword_match, line_match
from backend.analyzer.job_requirements import clean_job_text
from backend.analyzer.llm_client import call_llm
from backend.analyzer.llm_cost import calc_cost

logger = logging.getLogger("jobnavigator.rubric_ats_score")

PROVIDER = "openrouter"
MODEL = "google/gemma-4-31b-it"
# The models a caller may ask for by a short name, each with the provider that serves it.
GEMMA, HAIKU, HAIKU_ON_THE_CLAUDE_PLAN = "gemma", "haiku", "haiku_plan"
# The last one runs on the Claude Code login of a local backend: no OpenRouter credit is spent or needed.
MODELS = {GEMMA: (PROVIDER, MODEL), HAIKU: (PROVIDER, "anthropic/claude-haiku-5.5"),
          HAIKU_ON_THE_CLAUDE_PLAN: ("claude_code", "claude-haiku-5-5")}
# Eight reasons of a sentence each, eight picks, and three missing requirements.
MOST_REPLY_TOKENS = 700
MOST_WORDS_IN_A_REASON = 20
RESUME_MAX_CHARS = 24_000
POSTING_MAX_CHARS = 8_000
MOST_MISSING_REQUIREMENTS = 3
# How many times the model is asked. The middle pick of an odd number of answers needs no tie-break.
TIMES_ASKED = 3
# Changed whenever the rubric, the prompt or the arithmetic changes, so remembered scores are not reused.
VERSION = "4"
MOST_SCORES_REMEMBERED = 500


# ── The rubric ─────────────────────────────────────────────────────────────────
#
# Each option has the share it stands for. A share of None means the posting does not ask for that part,
# so the part is left out and its weight is shared among the others.

TITLE, CORE_SKILLS, OTHER_SKILLS, DUTIES = "title", "core_skills", "other_skills", "duties"
EXPERIENCE, QUALIFICATIONS, EVIDENCE, OCCUPATION = "experience", "qualifications", "evidence", "occupation"
KEYWORDS = "keywords"

HOW_MANY = {
    "all": ("All of them", 1.0),
    "most": ("Most: at least three quarters", 0.8),
    "about_half": ("About half: between a third and three quarters", 0.5),
    "few": ("A few: less than a third", 0.25),
    "none": ("None of them", 0.0),
}
NOT_STATED = {"not_stated": ("The posting does not ask for this", None)}

RUBRIC = {
    OCCUPATION: {
        "asks": "Is the candidate's career in the occupation this job is for? Go by job titles and main work, not tools.",
        "options": {
            "same": ("The same occupation", 1.0),
            "close": ("A closely related occupation: some main duties would be new", 0.8),
            "neighbouring": ("A neighbouring occupation: did part of this work in support of another job", 0.45),
            "different": ("A different occupation", 0.1),
        },
    },
    TITLE: {
        "asks": "How close is the resume's title, or its latest job title, to the posting's job title?",
        "options": {
            "exact": ("The same title, word for word", 1.0),
            "same_job": ("The same job in other words or at another seniority", 0.8),
            "related": ("A related job in the same field", 0.4),
            "different": ("A different job", 0.0),
        },
    },
    CORE_SKILLS: {
        "asks": "The few skills and tools this job is mainly about: how many has the candidate used in a role or project?",
        "options": HOW_MANY,
    },
    OTHER_SKILLS: {
        "asks": "The other skills and tools asked for, preferred ones included: how many used in a role or project?",
        "options": {**HOW_MANY, **NOT_STATED},
    },
    DUTIES: {
        "asks": "The duties the posting describes: how many has the candidate done?",
        "options": {**HOW_MANY, **NOT_STATED},
    },
    EXPERIENCE: {
        "asks": "Years and seniority asked for, counting only years in this kind of work. "
                "A seniority in the job title counts as asked for.",
        "options": {
            "meets": ("Meets or exceeds what is asked", 1.0),
            "slightly_below": ("Slightly below what is asked", 0.6),
            "far_below": ("Far below what is asked", 0.0),
            **NOT_STATED,
        },
    },
    QUALIFICATIONS: {
        "asks": "Degrees and certifications the posting requires. Leave out ones it only prefers.",
        "options": {
            "all": ("Has all of them", 1.0),
            "some": ("Has some of them", 0.5),
            "none": ("Has none of them", 0.0),
            **NOT_STATED,
        },
    },
    EVIDENCE: {
        "asks": "How well do the roles back up the skills the resume claims?",
        "options": {
            "strong": ("Roles show the skills in use, with results or scale", 1.0),
            "adequate": ("Roles show the skills in use, with few results", 0.7),
            "thin": ("Many skills are only listed or mentioned in passing", 0.3),
        },
    },
}

# How much each part counts in the total. The keyword match is the code's own part.
WEIGHT_OF_PART = {CORE_SKILLS: 0.30, DUTIES: 0.20, KEYWORDS: 0.15, OTHER_SKILLS: 0.10, EXPERIENCE: 0.10,
                  EVIDENCE: 0.05, TITLE: 0.05, QUALIFICATIONS: 0.05}
# An average cannot be sunk by one part, and a recruiter turns a candidate down for lacking the core skills.
# So the core skills also scale the whole score: (share of the core skills shown, share of the score kept).
SHARE_KEPT_BY_CORE_SKILLS_SHOWN = [(0.0, 0.4), (0.25, 0.6), (0.5, 0.85), (0.8, 1.0), (1.0, 1.0)]
MOST_CORE_SKILLS_NAMED = 5

SYSTEM_PROMPT = (
    "You grade a resume against a job posting. For every rubric question, write one reason of at most "
    f"{MOST_WORDS_IN_A_REASON} words naming what you saw, then pick exactly one option. "
    "Count work done in roles and projects; a skill that is only listed counts for little. "
    "The same work in different words counts. Where the posting offers alternatives, one is enough. "
    "Use only the two texts. "
    f"In core_skills_named, list the 3 to {MOST_CORE_SKILLS_NAMED} tools or platforms the job is mainly about, each "
    "as one product name from the posting (Kubernetes, not \"container orchestration\"). "
    f"In missing, name up to {MOST_MISSING_REQUIREMENTS} of the posting's most important requirements the resume "
    "does not show. Reply with the JSON object only."
)

REASON, OPTION, MISSING, CORE_SKILLS_NAMED = "reason", "option", "missing", "core_skills_named"


def _schema_of_the_reply() -> dict:
    """Strict JSON: for every question a reason and then one of its options, and the missing requirements."""
    properties = {CORE_SKILLS_NAMED: {"type": "array", "items": {"type": "string"}}}
    for question, details in RUBRIC.items():
        properties[question] = {
            "type": "object", "additionalProperties": False, "required": [REASON, OPTION],
            "properties": {REASON: {"type": "string"},
                           OPTION: {"type": "string", "enum": list(details["options"])}},
        }
    properties[MISSING] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


# ── The keyword match, in code ─────────────────────────────────────────────────

SHARE_OF_A_KEYWORD_ONLY_LISTED = 0.4


def keyword_check(resume_text: str, job_text: str) -> dict:
    """Where the résumé names each of the posting's keywords, as the posting writes them, and the share that earns.

    Returns {"used", "listed", "missing", "share"}. A keyword used in a role counts in full, one only listed
    as a skill counts for less, and one not named counts for nothing. share is None when the posting names none.
    """
    wanted = {name: mention["as_written"] for name, mention in keyword_match.skills_in(clean_job_text(job_text)).items()
              if mention["type"] == keyword_match.HARD_SKILL}
    in_the_work = set(keyword_match.skills_in(line_match._work_part_of(resume_text)))
    anywhere = set(keyword_match.skills_in(resume_text))
    used = [written for name, written in wanted.items() if name in in_the_work]
    listed = [written for name, written in wanted.items() if name in anywhere and name not in in_the_work]
    missing = [written for name, written in wanted.items() if name not in anywhere]
    share = (len(used) + SHARE_OF_A_KEYWORD_ONLY_LISTED * len(listed)) / len(wanted) if wanted else None
    return {"used": used, "listed": listed, "missing": missing, "share": None if share is None else round(share, 2)}


def core_skill_check(resume_text: str, named: list) -> dict:
    """Which of the skills the model called core the roles actually name, and the share that is.

    Returns {"named", "used", "not_used", "share"}; share is None when the model named none.
    """
    work = line_match._work_part_of(resume_text)
    skills_in_the_work = set(keyword_match.skills_in(work))
    named = [str(skill).strip() for skill in (named or []) if str(skill).strip()][:MOST_CORE_SKILLS_NAMED]
    used = [skill for skill in named if _is_used_in(skill, work, skills_in_the_work)]
    return {"named": named, "used": used, "not_used": [skill for skill in named if skill not in used],
            "share": round(len(used) / len(named), 2) if named else None}


def _is_used_in(skill: str, work: str, skills_in_the_work: set) -> bool:
    """By the dictionary's name for it when it has one (K8s is Kubernetes), else by the word itself."""
    known_as = set(keyword_match.skills_in(skill))
    return bool(known_as & skills_in_the_work) if known_as else line_match._names(work, skill)


# ── The years of work, in code ─────────────────────────────────────────────────
#
# Each role's line in the résumé text ends with its dates: "DevOps Engineer at Acme (Jun 2018 – Nov 2020".
# Overlapping roles are counted once. When the posting states a number of years, code answers the experience
# question; when it does not, the model's pick stands.

MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
A_MONTH_AND_YEAR = r"(?:([A-Za-z]{3})[a-z]*\.?\s+)?((?:19|20)\d{2})"
DATES_OF_A_ROLE = re.compile(A_MONTH_AND_YEAR + r"\s*[–—-]+\s*(?:(present|current|now|today)|" + A_MONTH_AND_YEAR + ")",
                             re.IGNORECASE)
YEARS_THE_POSTING_ASKS = re.compile(r"(\d{1,2})\s*(?:\+|plus)?\s*(?:(?:-|–|to)\s*\d{1,2}\s*\+?\s*)?(?:years?|yrs?)\b",
                                    re.IGNORECASE)
MOST_YEARS_A_POSTING_ASKS = 30
SHARE_OF_THE_YEARS_THAT_IS_SLIGHTLY_BELOW = 0.8


def years_of_work(resume_text: str, today: date | None = None):
    """The years the roles cover, from their dates, each month counted once. None when no role states its dates."""
    today = today or date.today()
    months_worked = set()
    for line in line_match._work_part_of(resume_text).splitlines():
        if line.lstrip().startswith("-"):
            continue
        for start_month, start_year, still_there, end_month, end_year in DATES_OF_A_ROLE.findall(line):
            first = _month_number(start_year, start_month, 1)
            last = _month_number(today.year, MONTHS[today.month - 1], 12) if still_there \
                else _month_number(end_year, end_month, 12)
            months_worked.update(range(first, last + 1))
    return round(len(months_worked) / 12, 1) if months_worked else None


def _month_number(year, month_name: str, month_when_not_stated: int) -> int:
    """Months counted from year 0, so a range of them can be walked."""
    name = (month_name or "").lower()[:3]
    month = MONTHS.index(name) + 1 if name in MONTHS else month_when_not_stated
    return int(year) * 12 + month


def years_the_posting_asks_for(job_text: str):
    """The largest number of years the posting states as a minimum, or None when it states none."""
    years = [int(number) for number in YEARS_THE_POSTING_ASKS.findall(line_match.posting_as_plain_text(job_text))]
    years = [number for number in years if 0 < number <= MOST_YEARS_A_POSTING_ASKS]
    return max(years) if years else None


def experience_from_the_dates(resume_text: str, job_text: str):
    """The experience option the dates settle, with the reason in words, or None when they cannot settle it."""
    worked, asked = years_of_work(resume_text), years_the_posting_asks_for(job_text)
    if worked is None or asked is None:
        return None
    if worked >= asked:
        option = "meets"
    elif worked >= SHARE_OF_THE_YEARS_THAT_IS_SLIGHTLY_BELOW * asked:
        option = "slightly_below"
    else:
        option = "far_below"
    return {OPTION: option, REASON: f"Counted from the dates: {worked:g} years in the roles, {asked} asked for."}


# ── What the model reads ───────────────────────────────────────────────────────

SKILLS_SECTION = "skills"


def _resume_without_its_skills_list(resume_text: str) -> str:
    """The résumé minus its skills section: the keyword check has read that, and it is a third of the tokens."""
    headings = list(line_match.HEADING_OF_THE_RESUME_TEXT.finditer(resume_text or ""))
    kept, position = [], 0
    for number, heading in enumerate(headings):
        if heading.group(1).strip().lower() == SKILLS_SECTION:
            end = headings[number + 1].start() if number + 1 < len(headings) else len(resume_text)
            kept.append(resume_text[position:heading.start()])
            position = end
    kept.append((resume_text or "")[position:])
    return "".join(kept).strip()


def _prompt(resume_text: str, job_text: str, keywords: dict) -> str:
    questions = []
    for question, details in RUBRIC.items():
        options = "; ".join(f"{option} = {meaning}" for option, (meaning, _share) in details["options"].items())
        questions.append(f"{question}: {details['asks']}\n  {options}")
    keyword_lines = (f"Used in roles: {', '.join(keywords['used']) or 'none'}\n"
                     f"Only listed as skills: {', '.join(keywords['listed']) or 'none'}\n"
                     f"Not in the resume: {', '.join(keywords['missing']) or 'none'}")
    posting = line_match.posting_as_plain_text(job_text)[:POSTING_MAX_CHARS]
    resume = _resume_without_its_skills_list(resume_text)[:RESUME_MAX_CHARS]
    return ("RUBRIC\n" + "\n".join(questions)
            + f"\n\nKEYWORD CHECK (already done, word for word)\n{keyword_lines}"
            + f"\n\nJOB POSTING\n<<<\n{posting}\n>>>\n\nRESUME\n<<<\n{resume}\n>>>")


# ── Asking the model ───────────────────────────────────────────────────────────

_scores_remembered = {}


async def ats_score(resume_text: str, job_text: str, provider: str = PROVIDER, model: str = MODEL) -> dict | None:
    """The score out of 100 with every part behind it, the model's reasons, and what the calls cost.

    Returns {"ats_score", "scorer", "model", "same_occupation", "core_skills_kept", "total_of_parts", "parts",
    "rubric": [rows], "keywords", "core_skills", "missing", "times_asked", "input_tokens", "output_tokens",
    "cost_usd"}. A rubric row is {"question", "asks", "option", "meaning", "share", "weight", "reason"}.
    Returns None when the model cannot be asked or none of its replies is usable.
    """
    remembered_as = _key_of(resume_text, job_text, provider, model)
    if remembered_as in _scores_remembered:
        return _scores_remembered[remembered_as]

    keywords = keyword_check(resume_text, job_text)
    prompt = _prompt(resume_text, job_text, keywords)
    replies = await asyncio.gather(*[_one_reply(prompt, provider, model) for _ in range(TIMES_ASKED)])
    replies = [reply for reply in replies if reply is not None]
    if not replies:
        return None

    all_answers = [reply["answers"] for reply in replies]
    answers = _middle_answers(all_answers)
    shares = _average_shares(all_answers)
    from_the_dates = experience_from_the_dates(resume_text, job_text)
    if from_the_dates:
        answers[EXPERIENCE] = from_the_dates
        shares[EXPERIENCE] = RUBRIC[EXPERIENCE]["options"][from_the_dates[OPTION]][1]
    core_skills = core_skill_check(resume_text, _core_skills_most_replies_name(all_answers))
    shares[OCCUPATION] = RUBRIC[OCCUPATION]["options"][answers[OCCUPATION][OPTION]][1]
    shares[CORE_SKILLS] = _core_skills_share(answers[CORE_SKILLS][OPTION], core_skills["share"])
    input_tokens = sum(reply["input_tokens"] for reply in replies)
    output_tokens = sum(reply["output_tokens"] for reply in replies)
    score = score_from_shares(shares, keywords["share"], core_skills["share"])
    result = {**score, "scorer": "rubric", "model": replies[0]["model"] or model,
              "rubric": _rubric_rows(answers, score["parts"]), "keywords": keywords, "core_skills": core_skills,
              "missing": [str(item) for item in (answers.get(MISSING) or [])][:MOST_MISSING_REQUIREMENTS],
              "times_asked": len(replies), "input_tokens": input_tokens, "output_tokens": output_tokens,
              "cost_usd": calc_cost(provider, model, input_tokens, output_tokens)}
    _remember(remembered_as, result)
    return result


async def _one_reply(prompt: str, provider: str, model: str) -> dict | None:
    """One answer to the rubric: {"answers", "model", "input_tokens", "output_tokens"}, or None when it is not usable."""
    try:
        reply = await call_llm(prompt, SYSTEM_PROMPT, max_tokens=MOST_REPLY_TOKENS, provider=provider, model=model,
                               response_schema=_schema_of_the_reply(), schema_name="ats_rubric", temperature=0)
    except Exception as error:
        logger.warning(f"A rubric call failed ({type(error).__name__}: {error})")
        return None
    answers = _answers_in(reply.get("text") or "")
    if answers is None:
        logger.warning(f"A rubric reply was not usable: {(reply.get('text') or '')[:200]!r}")
        return None
    usage = reply.get("usage") or {}
    return {"answers": answers, "model": reply.get("model"), "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0}


def _middle_answers(all_answers: list) -> dict:
    """For every question the middle pick of the replies, with the reason of a reply that made that pick.

    The missing requirements come from the reply that agrees with the most middle picks.
    """
    middle = {}
    for question, details in RUBRIC.items():
        order = list(details["options"])
        picks = sorted((answers[question][OPTION] for answers in all_answers), key=order.index)
        pick = picks[(len(picks) - 1) // 2]
        reason = next(answers[question].get(REASON) for answers in all_answers if answers[question][OPTION] == pick)
        middle[question] = {OPTION: pick, REASON: str(reason or "")}
    most_typical = max(all_answers, key=lambda answers: sum(answers[q][OPTION] == middle[q][OPTION] for q in RUBRIC))
    middle[MISSING] = most_typical.get(MISSING) or []
    return middle


# The two answers that scale the whole score are not averaged: one odd reply out of three would move the
# score by a quarter. They take the middle pick, which one odd reply cannot move.
PICKS_THAT_LEAVE_CORE_SKILLS_TO_THE_COUNT = ("all", "most")


def _core_skills_share(middle_pick: str, share_the_roles_name):
    """What the core skills earn: the count from the roles, unless the model sees a real gap the count misses.

    "All" and "most" are a hair apart to the model and six points apart in the score, so between those two
    the count decides. A pick of half or less is the model saying the roles fall short, and it can only lower.
    """
    from_the_pick = HOW_MANY[middle_pick][1]
    if share_the_roles_name is None:
        return from_the_pick
    if middle_pick in PICKS_THAT_LEAVE_CORE_SKILLS_TO_THE_COUNT:
        return share_the_roles_name
    return min(from_the_pick, share_the_roles_name)


def _average_shares(all_answers: list) -> dict:
    """For every question, the average of what the replies' picks stand for.

    A question most replies say the posting does not ask about is left out (None); otherwise the replies
    that did answer it are averaged.
    """
    shares = {}
    for question, details in RUBRIC.items():
        stated = [details["options"][answers[question][OPTION]][1] for answers in all_answers]
        answered = [share for share in stated if share is not None]
        most_say_not_asked = len(answered) <= len(stated) / 2
        shares[question] = None if most_say_not_asked else sum(answered) / len(answered)
    return shares


def _core_skills_most_replies_name(all_answers: list) -> list:
    """The core skills more than half of the replies name, in the order first named. All of one reply's, if none."""
    named_by = {}
    for answers in all_answers:
        for skill in {str(skill).strip() for skill in (answers.get(CORE_SKILLS_NAMED) or []) if str(skill).strip()}:
            named_by.setdefault(skill.lower(), [skill, 0])[1] += 1
    agreed = [skill for skill, times in named_by.values() if times > len(all_answers) / 2]
    return agreed or list(all_answers[0].get(CORE_SKILLS_NAMED) or [])


def _key_of(resume_text: str, job_text: str, provider: str, model: str) -> str:
    whole = "\x1f".join((VERSION, provider, model, resume_text or "", job_text or ""))
    return hashlib.sha256(whole.encode("utf-8")).hexdigest()


def _remember(key: str, result: dict) -> None:
    if len(_scores_remembered) >= MOST_SCORES_REMEMBERED:
        _scores_remembered.pop(next(iter(_scores_remembered)))
    _scores_remembered[key] = result


def _answers_in(reply_text: str) -> dict | None:
    """The model's answers, or None when a question is missing or its option is not one of that question's."""
    try:
        answers = json.loads(reply_text[reply_text.index("{"):reply_text.rindex("}") + 1])
    except ValueError:
        return None
    for question, details in RUBRIC.items():
        answer = answers.get(question) if isinstance(answers, dict) else None
        if not isinstance(answer, dict) or answer.get(OPTION) not in details["options"]:
            return None
    return answers


def _rubric_rows(answers: dict, parts: dict) -> list:
    """One row per question. The share is what the part earned, which for core skills the code may have lowered."""
    rows = []
    for question, details in RUBRIC.items():
        option = answers[question][OPTION]
        meaning, share = details["options"][option]
        earned = parts.get(question, share)
        rows.append({"question": question, "asks": details["asks"], "option": option, "meaning": meaning,
                     "share": None if earned is None else round(earned, 2), "weight": WEIGHT_OF_PART.get(question),
                     "reason": str(answers[question].get(REASON) or "")})
    return rows


# ── Scoring, in code ───────────────────────────────────────────────────────────

def score_from_picks(picks: dict, keyword_share, core_skills_used_share=None) -> dict:
    """The score when every question has one pick."""
    shares = {question: RUBRIC[question]["options"][pick][1] for question, pick in picks.items()}
    return score_from_shares(shares, keyword_share, core_skills_used_share)


def score_from_shares(shares: dict, keyword_share, core_skills_used_share=None) -> dict:
    """The score from what each question earned and the code's checks: parts by weight, then occupation and core skills."""
    parts = {question: share for question, share in shares.items() if question != OCCUPATION}
    parts[KEYWORDS] = keyword_share
    if core_skills_used_share is not None:
        parts[CORE_SKILLS] = min(parts[CORE_SKILLS], core_skills_used_share)
    stated = {part: share for part, share in parts.items() if share is not None}
    weight_in_play = sum(WEIGHT_OF_PART[part] for part in stated)
    total_of_parts = sum(WEIGHT_OF_PART[part] * share for part, share in stated.items()) / weight_in_play
    same_occupation = shares[OCCUPATION]
    core_skills_kept = _share_kept_for_core_skills(parts[CORE_SKILLS])
    return {"ats_score": round(100 * total_of_parts * same_occupation * core_skills_kept),
            "same_occupation": round(same_occupation, 2), "core_skills_kept": round(core_skills_kept, 2),
            "total_of_parts": round(total_of_parts, 3), "parts": parts}


def _share_kept_for_core_skills(share_shown: float) -> float:
    """The share of the score kept, rising evenly between the marked points so a small change never swings it."""
    points = SHARE_KEPT_BY_CORE_SKILLS_SHOWN
    for (lower, kept_at_lower), (upper, kept_at_upper) in zip(points, points[1:]):
        if share_shown <= upper:
            return kept_at_lower + (kept_at_upper - kept_at_lower) * (share_shown - lower) / (upper - lower)
    return points[-1][1]
