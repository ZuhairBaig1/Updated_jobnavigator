"""A keyword match score out of 100: hard skills 60, soft skills 15, other keywords 5, job title 10, degree 10.

It measures how closely a résumé's wording follows a posting, which is what to work on when tailoring. It is not
the fit score: it does not judge experience, seniority or whether the work was really done.
"""
import re

from backend.analyzer.job_requirements import clean_job_text
from backend.analyzer.keyword_match import HARD_SKILL, OTHER_KEYWORD, SOFT_SKILL, skills_in

POINTS = {HARD_SKILL: 60, SOFT_SKILL: 15, OTHER_KEYWORD: 5}
TITLE, DEGREE = "title", "degree"
# What each part is worth out of 100 when the posting has all five. A part the posting does not have is left
# out, and its points go to the others in proportion to their own weight.
PART_WEIGHTS = {**POINTS, TITLE: 10, DEGREE: 10}
LOWER_DEGREE_SHARE = 0.5

# Degree levels, lowest to highest, with the wordings that state each one.
DEGREE_LEVELS = [
    ("an Associate degree", r"\bassociate'?s?\s+degree|\bdiploma\b"),
    ("a Bachelor's degree", r"\bbachelor|\bb\.?\s?tech\b|\bb\.?e\.?\b(?=[\s,/)]|$)|\bb\.?sc?\.?\b|\bB\.?A\b|"
                            r"\bundergraduate|\bUG\b|\bany graduate\b|\bgraduate degree\b|\bdegree in\b"),
    ("a Master's degree", r"\bmaster'?s\b|\bmasters\b|\bmaster of\b|\bm\.?\s?tech\b|\bm\.?sc?\.?\b(?=[\s,/)]|$)|"
                          r"\bMBA\b|\bM\.?E\b|\bpost[- ]?graduate|\bPG\b"),
    ("a Ph.D.", r"\bph\.?\s?d\b|\bdoctorate|\bdoctoral"),
]


def _credit(times_in_posting: int, times_in_resume: int) -> float:
    """Nothing when never mentioned, at least half for one mention, all of it when mentioned as often as the posting."""
    if times_in_resume == 0:
        return 0.0
    if times_in_resume >= times_in_posting:
        return 1.0
    return 0.5 + 0.5 * times_in_resume / times_in_posting


def _keyword_rows(wanted: dict, shown: dict) -> list:
    rows = []
    for name, mention in wanted.items():
        times_in_resume = shown.get(name, {}).get("count", 0)
        rows.append({"keyword": mention["as_written"], "type": mention["type"],
                     "credit": round(_credit(mention["count"], times_in_resume), 3),
                     "resume": times_in_resume, "job": mention["count"]})
    return rows


def _share_for_type(rows: list, keyword_type: str) -> float | None:
    """The share of this type's keywords the résumé earns; None when the posting has no keywords of the type."""
    credits = [row["credit"] for row in rows if row["type"] == keyword_type]
    return sum(credits) / len(credits) if credits else None


def points_from_shares(shares: dict) -> tuple:
    """(points earned, points available) per part, with the parts the posting lacks left out and the rest scaled to 100."""
    present = {part: share for part, share in shares.items() if share is not None}
    total_weight = sum(PART_WEIGHTS[part] for part in present)
    if not total_weight:
        return {}, {}
    available = {part: round(100 * PART_WEIGHTS[part] / total_weight, 1) for part in present}
    earned = {part: round(100 * PART_WEIGHTS[part] * share / total_weight, 1) for part, share in present.items()}
    return earned, available


def _words_only(text: str) -> str:
    return re.sub(r"[^a-z0-9+#]+", " ", (text or "").lower()).strip()


def _title_is_in_resume(job_title: str, resume_text: str) -> bool | None:
    """Whether one line of the résumé holds every word of the posting's title, in any order; None with no title."""
    title_words = set(_words_only(job_title).split())
    if not title_words:
        return None
    return any(title_words <= set(_words_only(line).split()) for line in (resume_text or "").splitlines())


def _degree_levels_in(text: str) -> list:
    return [level for level, (_, pattern) in enumerate(DEGREE_LEVELS, start=1) if re.search(pattern, text or "", re.I)]


def _degree_share(resume_text: str, job_text: str) -> tuple:
    """(share earned or None when the posting asks for no degree, what the résumé shows, what the posting asks for)."""
    asked = _degree_levels_in(job_text)
    held = _degree_levels_in(resume_text)
    describe = lambda levels: DEGREE_LEVELS[max(levels) - 1][0] if levels else "no degree"
    if not asked:
        return None, describe(held), "no degree"
    # A posting naming two levels ("Bachelor's required, Master's preferred") is read as sitting between them.
    level_asked = sum(asked) / len(asked) if len(asked) > 1 else asked[0]
    if not held:
        return 0.0, "no degree", describe(asked)
    share = 1.0 if max(held) >= level_asked else LOWER_DEGREE_SHARE
    return share, describe(held), describe(asked)


def title_and_degree_shares(resume_text: str, posting: str, job_title: str) -> tuple:
    """({"title": share, "degree": share}, details): a share is None when the posting gives nothing to compare."""
    title_found = _title_is_in_resume(job_title, resume_text)
    degree_share, degree_held, degree_asked = _degree_share(resume_text, posting)
    shares = {TITLE: None if title_found is None else float(title_found), DEGREE: degree_share}
    return shares, {"title_found": title_found, "degree_in_resume": degree_held, "degree_asked": degree_asked}


def _counted_only_as_far_as_the_must_haves_are_met(shares: dict, must_haves_met: float | None) -> dict:
    """Every part but the hard skills earns its points only in step with the must-have hard skills shown.

    Soft skills, a matching title and a degree support the work. With half the must-have skills they earn at most
    half their points, and with none they earn none. A posting that names no hard skills leaves them as they are.
    """
    if must_haves_met is None:
        return shares
    return {part: share if part == HARD_SKILL or share is None else share * must_haves_met
            for part, share in shares.items()}


def scored(shares: dict, details: dict, rows: list, must_haves_met: float | None = None) -> dict:
    """must_haves_met is the share of must-have hard skills shown; left out, every hard skill counts as a must-have."""
    if must_haves_met is None:
        must_haves_met = shares.get(HARD_SKILL)
    shown_before_scaling = {part: None if share is None else round(share, 2) for part, share in shares.items()}
    earned, available = points_from_shares(_counted_only_as_far_as_the_must_haves_are_met(shares, must_haves_met))
    return {
        "match_score": round(sum(earned.values())) if earned else None,
        "match_breakdown": earned,
        "match_out_of": available,
        "match_details": {**details, "shown_before_scaling": shown_before_scaling,
                          "must_have_hard_skills_met": None if must_haves_met is None else round(must_haves_met, 2)},
        "keyword_table": rows,
    }


def match_score(resume_text: str, job_text: str, job_title: str = "") -> dict:
    """The score, how each part earned its points, and one row per keyword the posting asks for."""
    posting = clean_job_text(job_text)
    rows = _keyword_rows(skills_in(posting), skills_in(resume_text))
    shares = {keyword_type: _share_for_type(rows, keyword_type) for keyword_type in POINTS}
    other_shares, details = title_and_degree_shares(resume_text, posting, job_title)
    by_type_then_missing_first = lambda row: (list(POINTS).index(row["type"]), row["resume"] > 0, row["keyword"].lower())
    return scored({**shares, **other_shares}, details, sorted(rows, key=by_type_then_missing_first))
