"""The keyword match score with a decision model's judgment added to it.

Code still finds the posting's keywords. The model then says, for each one, how much the job depends on it and
how much evidence of it the résumé shows. A core tool the résumé only lists earns far less than one it shows in use.
"""
import logging

from backend.analyzer import decision_scorer
from backend.analyzer.job_requirements import clean_job_text
from backend.analyzer.keyword_match import HARD_SKILL, skills_in
from backend.analyzer.match_score import POINTS, match_score, scored, title_and_degree_shares
from backend.analyzer.prompt_fence import fence

logger = logging.getLogger("jobnavigator.judged_match")

IMPORTANCE_LEVELS = [
    ("Optional: the posting calls it preferred or a plus, or only mentions it in passing", 1.0),
    ("Required: the posting asks for it, as one requirement among several", 2.0),
    ("Core: it is what the job is mainly about, named in the title or stressed throughout the posting", 4.0),
]
IMPORTANCE_NAMES = ["optional", "required", "core"]

# Whether the résumé shows a keyword used the way this job expects, with the share of its points each level earns.
EVIDENCE_LEVELS = [
    ("Not in the resume", 0.0),
    ("Only listed in a skills section or summary, never shown in use", 0.3),
    ("Used, but for a different kind of work than this job expects, or as a side task in another kind of job", 0.5),
    ("Used for the kind of work this job expects, in at least one role", 0.85),
    ("Used for the kind of work this job expects, as a main part of the candidate's job", 1.0),
]
EVIDENCE_NAMES = ["missing", "only listed", "different use", "used as expected", "main work"]

IMPORTANCE_QUESTION = "How much does this job depend on {keyword}?"
EVIDENCE_QUESTION = (
    "The job posting expects {keyword} to be used for the work it describes. Does the resume show the candidate "
    "using {keyword} for that kind of work? A different wording of the same thing counts. The word appearing in "
    "the resume is not enough: judge what the candidate did with it against what this job needs."
)


def _questions(keywords: list, wording: str, levels: list) -> list:
    return [{"key": f"k{index}", "instructions": wording.format(keyword=keyword), "levels": levels}
            for index, keyword in enumerate(keywords)]


def _name_of_likeliest(answer: dict, levels: list, names: list) -> str:
    chosen = decision_scorer._most_likely_level(answer, levels)
    return names[[label for label, _ in levels].index(chosen)]


async def _judge(scorer: str, api_key: str, state: str, keywords: list, wording: str, levels: list) -> dict:
    answers, _ = await decision_scorer._ask_in_batches(scorer, api_key, state, _questions(keywords, wording, levels))
    return {keyword: answers[f"k{index}"] for index, keyword in enumerate(keywords)}


def _posting_state(job_text: str) -> str:
    return fence(job_text[:decision_scorer.JOB_MAX_CHARS], "JOB POSTING")


def _evidence_state(resume_text: str, job_text: str, must_haves: list) -> str:
    """What the model reads to judge evidence: the résumé, the posting, and the skills the job cannot do without."""
    listed = "\n".join(f"- {keyword}" for keyword in must_haves) or "- (none marked)"
    return "\n\n".join([
        fence(resume_text[:decision_scorer.RESUME_MAX_CHARS], "CANDIDATE RESUME"),
        _posting_state(job_text),
        fence(listed, "MUST-HAVE HARD SKILLS FOR THIS JOB"),
    ])


def _must_have_keywords(wanted: dict, importance: dict) -> list:
    return [mention["as_written"] for mention in wanted.values()
            if mention["type"] == HARD_SKILL
            and _name_of_likeliest(importance[mention["as_written"]], IMPORTANCE_LEVELS, IMPORTANCE_NAMES) != "optional"]


def _share_for_type(rows: list, keyword_type: str) -> float | None:
    """The share of this type earned, counting each keyword by its importance; None when the posting has none."""
    of_this_type = [row for row in rows if row["type"] == keyword_type]
    total_weight = sum(row["weight"] for row in of_this_type)
    if not total_weight:
        return None
    return sum(row["weight"] * row["credit"] for row in of_this_type) / total_weight


async def judged_match_score(scorer: str, resume_text: str, job_text: str, job_title: str = "") -> dict | None:
    """The match score with every keyword judged for importance and evidence; None when the model cannot be asked."""
    counted = match_score(resume_text, job_text, job_title)
    wanted = skills_in(clean_job_text(job_text))
    api_key = decision_scorer._find_api_key(scorer)
    if not wanted or not api_key:
        return None

    keywords = [mention["as_written"] for mention in wanted.values()]
    try:
        # Importance comes first: the evidence question is asked knowing which skills the job cannot do without.
        importance = await _judge(scorer, api_key, _posting_state(job_text), keywords,
                                  IMPORTANCE_QUESTION, IMPORTANCE_LEVELS)
        evidence = await _judge(scorer, api_key,
                                _evidence_state(resume_text, job_text, _must_have_keywords(wanted, importance)),
                                keywords, EVIDENCE_QUESTION, EVIDENCE_LEVELS)
    except Exception as error:
        logger.warning(f"Judged keyword match failed ({type(error).__name__}: {error})")
        return None

    rows = []
    for mention in wanted.values():
        keyword = mention["as_written"]
        rows.append({
            "keyword": keyword, "type": mention["type"],
            "importance": _name_of_likeliest(importance[keyword], IMPORTANCE_LEVELS, IMPORTANCE_NAMES),
            "evidence": _name_of_likeliest(evidence[keyword], EVIDENCE_LEVELS, EVIDENCE_NAMES),
            "weight": round(decision_scorer._share_earned(importance[keyword], IMPORTANCE_LEVELS), 2),
            "credit": round(decision_scorer._share_earned(evidence[keyword], EVIDENCE_LEVELS), 2),
        })

    shares = {keyword_type: _share_for_type(rows, keyword_type) for keyword_type in POINTS}
    other_shares, details = title_and_degree_shares(resume_text, clean_job_text(job_text), job_title)
    most_important_first = lambda row: (list(POINTS).index(row["type"]), -row["weight"], row["credit"])
    return {**scored({**shares, **other_shares}, details, sorted(rows, key=most_important_first),
                     must_haves_met=_must_have_hard_skills_met(rows)),
            "match_score_by_count": counted["match_score"]}


def _must_have_hard_skills_met(rows: list) -> float | None:
    """The share of the core and required hard skills the résumé shows; None when the posting has no hard skills."""
    hard_skills = [row for row in rows if row["type"] == HARD_SKILL]
    must_haves = [row for row in hard_skills if row["importance"] != "optional"] or hard_skills
    total_weight = sum(row["weight"] for row in must_haves)
    return sum(row["weight"] * row["credit"] for row in must_haves) / total_weight if total_weight else None
