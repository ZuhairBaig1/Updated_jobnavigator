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
import time
from datetime import date

from backend.analyzer import keyword_match, line_match
from backend.analyzer.job_requirements import clean_job_text
from backend.analyzer.llm_client import call_llm
from backend.analyzer.llm_cost import calc_cost
from backend.analyzer.llm_logger import log_llm_call

logger = logging.getLogger("jobnavigator.rubric_ats_score")

PROVIDER = "openrouter"
MODEL = "google/gemma-4-31b-it"
# The models a caller may ask for by a short name, each with the provider that serves it.
GEMMA, HAIKU, HAIKU_ON_THE_CLAUDE_PLAN = "gemma", "haiku", "haiku_plan"
LUNA_ON_THE_CHATGPT_PLAN = "luna_plan_high"
# The last two run on a login of a local backend (Claude Code, Codex): no OpenRouter credit is spent or needed.
MODELS = {GEMMA: (PROVIDER, MODEL), HAIKU: (PROVIDER, "anthropic/claude-haiku-5.5"),
          HAIKU_ON_THE_CLAUDE_PLAN: ("claude_code", "claude-haiku-5-5"),
          LUNA_ON_THE_CHATGPT_PLAN: ("codex_cli", "gpt-6-luna:high")}
# Seven reasons of a sentence each, seven picks, the duties sorted into two lists, and three missing requirements.
MOST_REPLY_TOKENS = 1000
MOST_WORDS_IN_A_REASON = 20
RESUME_MAX_CHARS = 24_000
POSTING_MAX_CHARS = 8_000
MOST_MISSING_REQUIREMENTS = 3
# How many times the model is asked. The middle pick of an odd number of answers needs no tie-break.
TIMES_ASKED = 3
# Changed whenever the rubric, the prompt or the arithmetic changes, so remembered scores are not reused.
VERSION = "9"
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

# The duties are not a pick. The model sorts each duty the posting describes into met or not met, and code
# counts them: picking between "all" and "most" by eye, it called a résumé's thin disaster-recovery work done.
DUTIES_MET, DUTIES_NOT_MET = "duties_met", "duties_not_met"
DUTIES_ASKS = "The duties the posting describes: how many do the roles show the candidate doing?"
COUNTED = "counted"
MOST_DUTIES_NAMED_IN_A_REASON = 4

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
    "In shown_another_way, for each keyword the keyword check gives as only listed or not in the resume: if a role "
    "shows the candidate using it under another name (a tool that is a kind of it, or a product built on it), give "
    "the keyword and, in shown_by, that name copied word for word from the role and nothing else. An answer whose "
    "shown_by is not written in a role is thrown away. Related work with no such name does not count. Leave out the rest. "
    "In duties_met and duties_not_met, sort every duty the posting describes, a few words each, by whether a role "
    "shows the candidate doing it. "
    f"In missing, name up to {MOST_MISSING_REQUIREMENTS} of the posting's most important requirements the resume "
    "does not show. Reply with the JSON object only."
)

REASON, OPTION, MISSING, CORE_SKILLS_NAMED = "reason", "option", "missing", "core_skills_named"
SHOWN_ANOTHER_WAY, KEYWORD, SHOWN_BY = "shown_another_way", "keyword", "shown_by"


def _schema_of_the_reply() -> dict:
    """Strict JSON: for every question a reason and then one of its options, and the missing requirements."""
    list_of_names = {"type": "array", "items": {"type": "string"}}
    properties = {CORE_SKILLS_NAMED: list_of_names, DUTIES_MET: list_of_names, DUTIES_NOT_MET: list_of_names}
    for question, details in RUBRIC.items():
        properties[question] = {
            "type": "object", "additionalProperties": False, "required": [REASON, OPTION],
            "properties": {REASON: {"type": "string"},
                           OPTION: {"type": "string", "enum": list(details["options"])}},
        }
    properties[SHOWN_ANOTHER_WAY] = {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": [KEYWORD, SHOWN_BY],
        "properties": {KEYWORD: {"type": "string"}, SHOWN_BY: {"type": "string"}}}}
    properties[MISSING] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


# ── The keyword match, in code ─────────────────────────────────────────────────

SHARE_OF_A_KEYWORD_ONLY_LISTED = 0.4


# A line that offers a choice ("at least one of Clojure, TypeScript, Python", "Terraform or similar") is one
# requirement, met by any one of the keywords it names. Counted keyword by keyword, a résumé with Python was
# charged for lacking Clojure and TypeScript, which the posting never asked for together.
A_CHOICE_ANYWHERE_IN_THE_LINE = re.compile(
    r"\b(at least one|one of|any of|either|such as|e\.g|for example|for instance|similar|equivalent)\b", re.IGNORECASE)
OR_BETWEEN_KEYWORDS = re.compile(r"\bor\b|/", re.IGNORECASE)
PLACES, USED, LISTED, MISSING_KEYWORD = ("used", "listed", "missing"), "used", "listed", "missing"
WORTH_OF_A_KEYWORD = {USED: 1.0, LISTED: SHARE_OF_A_KEYWORD_ONLY_LISTED, MISSING_KEYWORD: 0.0}


def keyword_check(resume_text: str, job_text: str) -> dict:
    """Where the résumé names each keyword the posting asks for, as the posting writes them, and the share that earns.

    Returns {"used", "listed", "missing", "alternatives_not_needed", "share"}. A keyword used in a role counts
    in full, one only listed as a skill counts for less, and one not named counts for nothing. Keywords a line
    offers as a choice are one requirement, judged by the best of them; the others it did not need are listed
    apart. share is None when the posting names no keyword.
    """
    in_the_work = set(keyword_match.skills_in(line_match._work_part_of(resume_text)))
    anywhere = set(keyword_match.skills_in(resume_text))

    def place_of(name: str) -> str:
        return USED if name in in_the_work else LISTED if name in anywhere else MISSING_KEYWORD

    requirements, not_needed = [], []
    for requirement in keyword_requirements(job_text):
        best = max(requirement, key=lambda name: WORTH_OF_A_KEYWORD[place_of(name)])
        place = place_of(best)
        if place == MISSING_KEYWORD:
            requirements.append({"label": " or ".join(requirement.values()), "place": place})
        else:
            requirements.append({"label": requirement[best], "place": place})
            not_needed += [written for name, written in requirement.items()
                           if name != best and place_of(name) == MISSING_KEYWORD]
    return _keywords_summed_up(requirements, not_needed, shown_another_way=[])


def _keywords_summed_up(requirements: list, not_needed: list, shown_another_way: list) -> dict:
    """The keyword check as its lists and its share, from each requirement's label and place."""
    found = {place: list(dict.fromkeys(r["label"] for r in requirements if r["place"] == place)) for place in PLACES}
    everything_placed = found[USED] + found[LISTED] + found[MISSING_KEYWORD]
    worth = [WORTH_OF_A_KEYWORD[requirement["place"]] for requirement in requirements]
    return {**found,
            "alternatives_not_needed": [w for w in dict.fromkeys(not_needed) if w not in everything_placed],
            "shown_another_way": shown_another_way,
            "share": round(sum(worth) / len(worth), 2) if worth else None,
            "requirements": requirements}


# A keyword the résumé only lists, or does not name, may still be shown in a role under another name:
# Terraform is Infrastructure as Code, and EKS is Kubernetes. Word matching cannot see that, so the model
# says which of those keywords a role shows and what the role calls it. It counts when most replies agree.
#
# The model has to quote the role, and a quote that is not in a role is thrown away. Without that, one reply
# gave "Amazon EKS" as its reason on a résumé that never says EKS, and the keyword was counted all the same.
# The model still makes the judgement; code only refuses a reason the résumé does not contain.
SHORTEST_NAME_QUOTED = 2
def with_keywords_shown_another_way(keywords: dict, all_answers: list, resume_text: str) -> dict:
    """The keyword check with every requirement most replies say a role shows under another name counted as used.

    A reply's answer counts only when the name it quotes is written in a role.
    """
    work = _as_plain_words(line_match._work_part_of(resume_text))
    said_by = {}
    for answers in all_answers:
        named = {}
        for item in answers.get(SHOWN_ANOTHER_WAY) or []:
            if not isinstance(item, dict) or not str(item.get(KEYWORD) or "").strip():
                continue
            quoted = names_written_in_the_roles(str(item.get(SHOWN_BY) or ""), work)
            if quoted:
                named.setdefault(str(item[KEYWORD]).strip().lower(), quoted)
            else:
                logger.warning("Rubric: a reply said %r is shown by %r, which no role says — left out",
                               item.get(KEYWORD), item.get(SHOWN_BY))
        for keyword, shown_by in named.items():
            said_by.setdefault(keyword, []).append(shown_by)
    agreed = {keyword: names[0] for keyword, names in said_by.items() if len(names) > len(all_answers) / 2}

    requirements, shown = [], []
    for requirement in keywords["requirements"]:
        keyword = _keyword_agreed_for(requirement, agreed)
        if keyword:
            shown.append({KEYWORD: requirement["label"], SHOWN_BY: agreed[keyword]})
            requirement = {**requirement, "place": USED}
        requirements.append(requirement)
    return _keywords_summed_up(requirements, keywords["alternatives_not_needed"], shown)


def names_written_in_the_roles(quoted: str, work_as_plain_words: str) -> str:
    """The part of what the model quoted that a role really says, or "" when none of it is there.

    A reply may quote several names at once ("EKS, AKS and GKE"); the ones a role writes are kept.
    """
    whole = _as_plain_words(quoted)
    if len(whole) >= SHORTEST_NAME_QUOTED and _is_written_in(whole, work_as_plain_words):
        return quoted.strip()
    pieces = [piece.strip() for piece in re.split(r",|;|\band\b|\bor\b|/", quoted) if piece.strip()]
    found = [piece for piece in pieces
             if len(_as_plain_words(piece)) >= SHORTEST_NAME_QUOTED and _is_written_in(_as_plain_words(piece), work_as_plain_words)]
    return ", ".join(found)


def _as_plain_words(text: str) -> str:
    """Lower case, with every run of spaces and punctuation as one space, so a quote is compared on its words."""
    return " ".join(re.findall(r"[a-z0-9+#.]+", (text or "").lower()))


def _is_written_in(words: str, text: str) -> bool:
    return f" {words} " in f" {text} "


def _keyword_agreed_for(requirement: dict, agreed: dict):
    """The agreed keyword that names this requirement, when the résumé does not already use it in a role."""
    if requirement["place"] == USED:
        return None
    its_keywords = [part.strip().lower() for part in requirement["label"].split(" or ")]
    return next((keyword for keyword in agreed if keyword in its_keywords), None)


def keyword_requirements(job_text: str) -> list:
    """The posting's keywords as requirements: each a {standard name: as written} of one keyword, or of several
    when a line offers them as a choice. A keyword asked for alone anywhere is a requirement on its own too."""
    alone, choices = {}, []
    for line in line_match.posting_lines(clean_job_text(job_text)):
        keywords = {name: mention["as_written"] for name, mention in keyword_match.skills_in(line).items()
                    if mention["type"] == keyword_match.HARD_SKILL}
        if len(keywords) > 1 and _offers_its_keywords_as_a_choice(line, list(keywords.values())):
            choices.append(keywords)
        else:
            alone.update(keywords)
    asked_alone = [{name: written} for name, written in alone.items()]
    # A choice every keyword of which is also asked for alone adds nothing.
    choices_still_open = [choice for choice in choices if not set(choice) <= set(alone)]
    return asked_alone + _without_repeats(choices_still_open)


def _offers_its_keywords_as_a_choice(line: str, keywords_as_written: list) -> bool:
    """A choice is worded as one ("such as", "at least one"), or has an "or" among the keywords themselves.

    An "or" elsewhere in the line ("applications or services, using AWS and Docker") offers no choice of keywords.
    """
    if A_CHOICE_ANYWHERE_IN_THE_LINE.search(line):
        return True
    places = [line.lower().find(written.lower()) for written in keywords_as_written]
    places = [place for place in places if place >= 0]
    return bool(places) and bool(OR_BETWEEN_KEYWORDS.search(line[min(places):max(places)]))


def _without_repeats(choices: list) -> list:
    kept = []
    for choice in choices:
        if not any(set(choice) == set(other) for other in kept):
            kept.append(choice)
    return kept


def core_skill_check(resume_text: str, named: list, shown_another_way: list = ()) -> dict:
    """Which of the skills the model called core the roles actually name, and the share that is.

    A core skill the replies agreed a role shows under another name (EKS for Kubernetes) counts as named.
    Returns {"named", "used", "not_used", "share"}; share is None when the model named none.
    """
    work = line_match._work_part_of(resume_text)
    skills_in_the_work = set(keyword_match.skills_in(work))
    shown_under_another_name = {part.strip().lower() for item in shown_another_way
                                for part in str(item[KEYWORD]).split(" or ")}
    named = [str(skill).strip() for skill in (named or []) if str(skill).strip()][:MOST_CORE_SKILLS_NAMED]
    used = [skill for skill in named
            if _is_used_in(skill, work, skills_in_the_work) or skill.lower() in shown_under_another_name]
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


# ── A result in the roles, in code ─────────────────────────────────────────────
#
# "Strong" evidence means roles that show results or scale. The model gave it to a résumé whose only results
# were "reduced deployment failures" and "lower MTTR", with no figure behind either. So strong evidence needs
# at least one number in a bullet: a percentage, an amount of money, a multiple, or a count of something.

A_PERCENTAGE_AN_AMOUNT_OR_A_MULTIPLE = re.compile(r"\d\s?%|[$€£₹]\s?\d|\b\d+(?:\.\d+)?x\b", re.IGNORECASE)
A_COUNT_OF_SOMETHING = re.compile(r"(?<![\w./-])(\d{1,3}(?:,\d{3})+|\d+)\+?\s+([a-z]{3,})")
WORDS_THAT_ARE_NOT_A_THING_COUNTED = {"and", "the", "for", "with", "into", "from", "that", "this", "are", "was", "were"}
NOT_STRONG = "adequate"
NO_NUMBER_NOTE = "No figure is given in any role, so this cannot count as strong."


def roles_state_a_result_in_numbers(resume_text: str) -> bool:
    """Whether any bullet under a role gives a figure: 30%, $2M, 3x, or a count such as "40 teams"."""
    for line in line_match._work_part_of(resume_text).splitlines():
        if not line.lstrip().startswith("-"):
            continue
        if A_PERCENTAGE_AN_AMOUNT_OR_A_MULTIPLE.search(line):
            return True
        if any(_is_a_count(line, match) for match in A_COUNT_OF_SOMETHING.finditer(line)):
            return True
    return False


def _is_a_count(line: str, match) -> bool:
    """"40 teams" is a count. "Java 17 applications" is a version, and "2019 projects" is a year."""
    number, thing = match.group(1).replace(",", ""), match.group(2)
    words_before = line[:match.start()].lstrip("-•* ").split()
    word_before = words_before[-1] if words_before else ""
    # The verb that opens a bullet is capitalised too ("Supported 40 teams"), and is no product name.
    is_a_name = word_before[:1].isupper() and len(words_before) > 1
    names_a_version = is_a_name or any(character.isdigit() for character in word_before)
    is_a_year = len(number) == 4 and number[:2] in ("19", "20")
    return not names_a_version and not is_a_year and thing not in WORDS_THAT_ARE_NOT_A_THING_COUNTED


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
    if keywords.get("alternatives_not_needed"):
        keyword_lines += ("\nAlternatives the posting offered and the resume did not need: "
                          + ", ".join(keywords["alternatives_not_needed"]))
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
    "rubric": [rows], "keywords", "core_skills", "duties", "missing", "times_asked", "remembered",
    "input_tokens", "output_tokens", "thinking_tokens", "cost_usd"}. thinking_tokens are part of output_tokens.
    A rubric row is {"question", "asks", "option", "meaning", "option_share", "share", "weight", "reason"}:
    option_share is what the option shown stands for, share is what the part earned over all the replies.
    Returns None when the model cannot be asked or none of its replies is usable.
    """
    remembered_as = _key_of(resume_text, job_text, provider, model)
    if remembered_as in _scores_remembered:
        return {**_scores_remembered[remembered_as], "remembered": True}

    keywords = keyword_check(resume_text, job_text)
    prompt = _prompt(resume_text, job_text, keywords)
    replies = await asyncio.gather(*[_one_reply(prompt, provider, model) for _ in range(TIMES_ASKED)])
    replies = [reply for reply in replies if reply is not None]
    if not replies:
        return None

    all_answers = [reply["answers"] for reply in replies]
    keywords = with_keywords_shown_another_way(keywords, all_answers, resume_text)
    answers = _middle_answers(all_answers)
    shares = _average_shares(all_answers)
    from_the_dates = experience_from_the_dates(resume_text, job_text)
    if from_the_dates:
        answers[EXPERIENCE] = from_the_dates
        shares[EXPERIENCE] = RUBRIC[EXPERIENCE]["options"][from_the_dates[OPTION]][1]
    duties = _duties_counted(all_answers)
    shares[DUTIES] = duties["share"]
    if not roles_state_a_result_in_numbers(resume_text):
        _hold_the_evidence_below_strong(answers, shares)
    core_skills = core_skill_check(resume_text, _core_skills_most_replies_name(all_answers),
                                   keywords["shown_another_way"])
    shares[OCCUPATION] = RUBRIC[OCCUPATION]["options"][answers[OCCUPATION][OPTION]][1]
    shares[CORE_SKILLS] = _core_skills_share(answers[CORE_SKILLS][OPTION], core_skills["share"])
    input_tokens = sum(reply["input_tokens"] for reply in replies)
    output_tokens = sum(reply["output_tokens"] for reply in replies)
    thinking_tokens = sum(reply["thinking_tokens"] for reply in replies)
    score = score_from_shares(shares, keywords["share"], core_skills["share"])
    result = {**score, "scorer": "rubric", "model": replies[0]["model"] or model,
              "rubric": _rubric_rows(answers, score["parts"], duties), "keywords": keywords,
              "core_skills": core_skills, "duties": duties, "remembered": False,
              "missing": [str(item) for item in (answers.get(MISSING) or [])][:MOST_MISSING_REQUIREMENTS],
              "times_asked": len(replies), "input_tokens": input_tokens, "output_tokens": output_tokens,
              "thinking_tokens": thinking_tokens,
              "cost_usd": calc_cost(provider, model, input_tokens, output_tokens)}
    _remember(remembered_as, result)
    return result


async def _one_reply(prompt: str, provider: str, model: str) -> dict | None:
    """One answer to the rubric: {"answers", "model", "input_tokens", "output_tokens", "thinking_tokens"}, or None
    when it is not usable. thinking_tokens are part of output_tokens."""
    started = time.monotonic()
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
    log_llm_call("ats_score", provider, model, usage, duration_ms=int(1000 * (time.monotonic() - started)))
    # Input the provider served from its cache is input all the same.
    input_tokens = sum(usage.get(key, 0) or 0 for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
    return {"answers": answers, "model": reply.get("model"), "input_tokens": input_tokens,
            "output_tokens": usage.get("output_tokens", 0) or 0, "thinking_tokens": usage.get("thinking_tokens", 0) or 0}


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


def _share_of_duties_met(answers: dict):
    """In one reply: the share of the duties it listed that it called met. None when it listed none."""
    met, not_met = len(answers.get(DUTIES_MET) or []), len(answers.get(DUTIES_NOT_MET) or [])
    return met / (met + not_met) if met + not_met else None


def _duties_counted(all_answers: list) -> dict:
    """The duties over all the replies: {"share", "met", "not_met"}.

    share is the average of the replies that listed any duty, or None when most listed none. The two lists
    are those of the reply whose share is the middle one.
    """
    counted = sorted(((_share_of_duties_met(answers), number) for number, answers in enumerate(all_answers)
                      if _share_of_duties_met(answers) is not None))
    if len(counted) <= len(all_answers) / 2:
        return {"share": None, "met": [], "not_met": []}
    middle = all_answers[counted[(len(counted) - 1) // 2][1]]
    return {"share": round(sum(share for share, _number in counted) / len(counted), 2),
            "met": [str(duty) for duty in middle.get(DUTIES_MET) or []],
            "not_met": [str(duty) for duty in middle.get(DUTIES_NOT_MET) or []]}


def _hold_the_evidence_below_strong(answers: dict, shares: dict) -> None:
    """With no figure in any role, the evidence earns no more than the option below strong, and says why."""
    most_it_can_earn = RUBRIC[EVIDENCE]["options"][NOT_STRONG][1]
    if shares[EVIDENCE] <= most_it_can_earn:
        return
    shares[EVIDENCE] = most_it_can_earn
    answers[EVIDENCE] = {OPTION: NOT_STRONG, REASON: NO_NUMBER_NOTE}


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


def _rubric_rows(answers: dict, parts: dict, duties: dict) -> list:
    """One row per part. The share is what the part earned, which code may have lowered or averaged."""
    rows = []
    for question, details in RUBRIC.items():
        option = answers[question][OPTION]
        meaning, share = details["options"][option]
        earned = parts.get(question, share)
        rows.append({"question": question, "asks": details["asks"], "option": option, "meaning": meaning,
                     "option_share": share, "share": None if earned is None else round(earned, 2),
                     "weight": WEIGHT_OF_PART.get(question), "reason": str(answers[question].get(REASON) or "")})
    rows.insert(_place_of_the_duties_row(rows), _duties_row(duties))
    return rows


def _place_of_the_duties_row(rows: list) -> int:
    """Straight after the two skills rows, where it has always been shown."""
    return 1 + max(number for number, row in enumerate(rows) if row["question"] in (CORE_SKILLS, OTHER_SKILLS))


def _duties_row(duties: dict) -> dict:
    met, not_met = duties["met"], duties["not_met"]
    if duties["share"] is None:
        meaning, reason, shown = "The posting describes no duties", "", None
    else:
        meaning = f"{len(met)} of {len(met) + len(not_met)} duties"
        shown = round(len(met) / (len(met) + len(not_met)), 2)
        reason = ("Not shown in a role: " + "; ".join(not_met[:MOST_DUTIES_NAMED_IN_A_REASON]) if not_met
                  else "Every duty the posting describes is shown in a role.")
    return {"question": DUTIES, "asks": DUTIES_ASKS, "option": COUNTED, "meaning": meaning, "option_share": shown,
            "share": duties["share"], "weight": WEIGHT_OF_PART[DUTIES], "reason": reason}


# ── Scoring, in code ───────────────────────────────────────────────────────────

def score_from_picks(picks: dict, keyword_share, core_skills_used_share=None, duties_share=None) -> dict:
    """The score when every question has one pick and the duties have been counted."""
    shares = {question: RUBRIC[question]["options"][pick][1] for question, pick in picks.items()}
    shares[DUTIES] = duties_share
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
