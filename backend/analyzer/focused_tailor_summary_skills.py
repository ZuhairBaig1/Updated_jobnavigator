"""The summary and the skills of a focused or complete copy, kept only as far as the base résumé supports them (step 3)."""
import logging
import re

from backend.analyzer import resume_schema as rs
from backend.analyzer.focused_tailor_basics import _bullets_of, _projects, _roles
from backend.analyzer.focused_tailor_checks import (_DETAIL, _adds_words_to, _is_stated, _is_supported,
                                                    _with_the_unsupported_items_taken_out, unsupported_in)

logger = logging.getLogger("jobnavigator.focused_tailor")

# A full stop ends a sentence unless it closes a short title such as "Sr." or "Jr.".
_SENTENCE_END = re.compile(r"(?<![ (][A-Z][a-z]\.)(?<![ (][A-Z][a-z]{2}\.)(?<=[.!?])\s+(?=[A-Z0-9])")


# ── The summary ─────────────────────────────────────────────────────────────────

def focused_summary(reply: dict, base: dict, base_text: str, vocabulary: set, other_names: dict) -> str:
    """The summary written for the posting, without what the résumé does not state; the base's own when nothing is left."""
    written = rs._text((reply or {}).get("professional_summary"))
    sentences = [sentence.strip() for sentence in _SENTENCE_END.split(written) if sentence.strip()]
    prose = _prose_of_the_resume(base)
    kept = []
    for sentence in sentences:
        unsupported = unsupported_in(sentence, base_text, vocabulary, other_names, prose=prose)
        if not unsupported:
            kept.append(sentence)
            continue
        # One unsupported item in a list should cost that item, not the whole sentence and its keywords.
        shorter = _with_the_unsupported_items_taken_out(sentence, unsupported, base_text, vocabulary, other_names, prose)
        if shorter:
            logger.warning("Focused tailor: the summary listed %s, which the résumé does not — taken out of the list",
                           ", ".join(unsupported[:6]))
            kept.append(shorter)
        else:
            logger.warning("Focused tailor: a summary sentence stated %s, which the résumé does not — left out",
                           ", ".join(unsupported[:6]))
    return " ".join(kept) or rs._text(base.get("summary"))


def _prose_of_the_resume(base: dict) -> str:
    """The base's summary and every bullet: the sentences a number in the summary may come from."""
    sentences = [rs._text(base.get("summary"))]
    for entry in _roles(base) + _projects(base):
        sentences += _bullets_of(entry)
    return " ".join(sentences)


# ── The skills ──────────────────────────────────────────────────────────────────

def focused_skills(reply: dict, base: dict, base_text: str, other_names: dict, jd_text: str = "") -> dict:
    """The skills chosen for the posting, each one the résumé has, under the posting's name where it is the same thing."""
    base_skills = base.get("skills") or {}
    base_terms = [term.strip() for value in base_skills.values() for term in rs._split_skills(value) if term.strip()]
    printed, chosen = [], {}
    for label, value in rs._skills_map((reply or {}).get("technical_skills")).items():
        kept = []
        for skill in rs._split_skills(value):
            supported = _skill_as_the_resume_supports_it(skill, base_text, base_terms, other_names)
            if supported and not _is_already_printed(supported, printed):
                kept.append(supported)
                printed.append(supported)
        if kept:
            chosen[label] = ", ".join(kept)
    if not chosen:
        return dict(base_skills)
    return _with_the_listed_skills_the_posting_names(chosen, printed, base_skills, jd_text)


def _is_already_printed(skill: str, printed: list) -> bool:
    for seen in printed:
        if rs._same_skill(skill, seen) or rs._squash(skill) == rs._squash(seen):
            return True
    return False


def _skill_as_the_resume_supports_it(skill: str, base_text: str, base_terms: list, other_names: dict) -> str:
    """How one skill of the reply may be printed: as written, under the résumé's own name, or not at all ("")."""
    detail = _DETAIL.search(skill)
    name = _DETAIL.sub("", skill).strip() if detail else skill.strip()
    if not name:
        return ""
    if not _is_supported(name, base_text, other_names) and not any(rs._same_skill(name, term) for term in base_terms):
        return _resume_name_it_narrows(name, base_terms)
    if not detail:
        return name
    members = []
    for member in detail.group(1).split(","):
        member = member.strip()
        if member and _is_supported(member, base_text, other_names):
            members.append(member)
    return f"{name} ({', '.join(members)})" if members else name


def _resume_name_it_narrows(name: str, base_terms: list) -> str:
    """The résumé's own name when `name` is it with words added (a narrower thing), longest first; else ""."""
    for term in sorted(base_terms, key=len, reverse=True):
        if _adds_words_to(name, term):
            return term
    return ""


def _with_the_listed_skills_the_posting_names(chosen: dict, printed: list, base_skills: dict, jd_text: str) -> dict:
    """Adds any skill the résumé lists and the posting names that the reply left out of its skills.

    Each goes beside the skills it sat with in the résumé, or into the last group when none of those were kept.
    """
    for label, value in base_skills.items():
        if "certif" in str(label).lower():
            continue
        terms = [term.strip() for term in rs._split_skills(value) if term.strip()]
        for term in terms:
            name = _DETAIL.sub("", term).strip()
            if len(rs._squash(name)) < 2 or not _is_stated(name, jd_text):
                continue
            if any(rs._same_skill(name, seen) or _is_stated(name, seen) for seen in printed):
                continue
            home = _group_holding_any_of(terms, chosen)
            chosen[home] = f"{chosen[home]}, {name}"
            printed.append(name)
    return chosen


def _group_holding_any_of(terms: list, chosen: dict) -> str:
    """The first chosen group holding one of these skills; the last group when none does."""
    for group, skills in chosen.items():
        for sibling in terms:
            if any(rs._same_skill(sibling, kept) for kept in rs._split_skills(skills)):
                return group
    return list(chosen)[-1]


def complete_skills(reply: dict, base: dict, base_text: str, other_names: dict, jd_text: str = "") -> dict:
    """Every skill of the résumé, grouped the reply's way so that what the posting names is read first.

    The reply's groups are checked as a focused copy's are, then each skill it left out goes back beside its
    old neighbours. A skill printed under the posting's name ("MSSQL") is not printed again under the résumé's.
    """
    chosen = focused_skills(reply, base, base_text, other_names, jd_text)
    base_skills = base.get("skills") or {}
    if chosen == dict(base_skills):
        return chosen
    printed = [skill for value in chosen.values() for skill in rs._split_skills(value)]
    renamed = _resume_names_printed_under_the_postings(other_names, printed)
    still_to_place = {}
    for label, value in base_skills.items():
        left = ", ".join(term for term in rs._split_skills(value) if rs._squash(term) not in renamed)
        if left:
            still_to_place[label] = left
    return rs._merge_skills(chosen, still_to_place)


def _resume_names_printed_under_the_postings(other_names: dict, printed: list) -> set:
    """The résumé's names (squashed) whose posting name is already printed among the skills."""
    renamed = set()
    for posting_name, resume_name in other_names.items():
        if any(_is_stated(posting_name, skill) for skill in printed):
            renamed.add(rs._squash(resume_name))
    return renamed
