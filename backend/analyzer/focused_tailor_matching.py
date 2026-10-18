"""Step 1 of a focused or complete copy: reading the reply's matching of the posting to the résumé.

The reply lists what the posting asks for, how much it wants each thing, and where the résumé shows it. This keeps
only what can be checked: a pointer must lead to a bullet or a skills line the résumé has, and the posting's name
for something the résumé names differently must truly be the same thing.
"""
import re

from backend.analyzer import resume_schema as rs
from backend.analyzer.focused_tailor_basics import (FROM_A_JOB, FROM_A_PROJECT, FROM_NOWHERE, FROM_THE_SKILLS,
                                                    MOST_WORDS_IN_A_KEYWORD)
from backend.analyzer.focused_tailor_checks import _KEYWORD_PARTS, _adds_words_to, _is_stated

# A reply's pointer at a bullet of a job or a project: "2.5" is its bullet 5 of job (or project) 2.
_ROLE_DOT_BULLET = re.compile(r"^\s*(\d+)\s*\.\s*(\d+)\s*$")


def other_names_allowed(reply: dict, base_text: str, jd_text: str) -> dict:
    """The posting's name for something the résumé names differently, as {posting name: résumé name}.

    The model proposes each pair. One is kept only when the résumé states its own name, the posting states
    the other, and the posting's name is not simply the résumé's with words added.
    """
    allowed = {}
    for pair in (reply or {}).get("same_thing_other_name") or []:
        if not isinstance(pair, dict):
            continue
        posting_name, resume_name = rs._text(pair.get("posting_name")), rs._text(pair.get("resume_name"))
        if not posting_name or not resume_name or rs._squash(posting_name) == rs._squash(resume_name):
            continue
        if not _is_stated(resume_name, base_text) or not _is_stated(posting_name, jd_text):
            continue
        if _adds_words_to(posting_name, resume_name):
            continue
        allowed[posting_name] = resume_name
    return allowed


def requirements_matched(reply: dict, roles: list, projects: list = (), skill_lines: int = 0) -> list:
    """The reply's matching of the posting to the résumé, one entry a keyword.

    The reply writes each as {"name", "kind", "has", "source", "by"}; each comes back as
    {"requirement", "importance", "shown_by", "resume_has_it", "source", "found_at"}.
    `found_at` is where the source shows it: (job, bullet), (project, bullet) or a skills line number.
    `shown_by` is the same for a job and empty otherwise, since only a job's bullets are chosen by what they show.
    """
    matched = []
    for entry in (reply or {}).get("requirements") or []:
        if not isinstance(entry, dict) or not rs._text(entry.get("name")):
            continue
        source = _source_of(entry)
        found_at = _places_it_points_at(entry.get("by"), source, roles, projects, skill_lines)
        shown_by = found_at if source == FROM_A_JOB else []
        resume_has_it = bool(entry.get("has")) or bool(found_at)
        for name in _one_keyword_each(rs._text(entry["name"])):
            matched.append({"requirement": name, "importance": entry.get("kind"), "shown_by": shown_by,
                            "resume_has_it": resume_has_it, "source": source, "found_at": found_at})
    return matched


def _source_of(entry: dict) -> str:
    """Where the reply says the résumé shows a requirement; a reply that names no source points at jobs, as before."""
    source = entry.get("source")
    return source if source in (FROM_A_JOB, FROM_A_PROJECT, FROM_THE_SKILLS, FROM_NOWHERE) else FROM_A_JOB


def _places_it_points_at(pointers, source: str, roles: list, projects: list, skill_lines: int) -> list:
    """The reply's pointers that exist in the source it named; any other is dropped."""
    if source == FROM_A_JOB:
        return _bullets_it_points_at(pointers, roles)
    if source == FROM_A_PROJECT:
        return _bullets_it_points_at(pointers, projects)
    if source == FROM_THE_SKILLS:
        return _skill_lines_it_points_at(pointers, skill_lines)
    return []


def _bullets_it_points_at(pointers, entries: list) -> list:
    """The reply's "entry.bullet" pointers into jobs or projects as (entry number, bullet number);
    one at an entry or bullet the résumé lacks is dropped."""
    pointed_at = []
    for pointer in pointers or []:
        written = _ROLE_DOT_BULLET.match(str(pointer))
        if not written:
            continue
        entry_number, bullet_number = int(written.group(1)), int(written.group(2))
        entry_exists = 1 <= entry_number <= len(entries)
        if entry_exists and 1 <= bullet_number <= len(entries[entry_number - 1].get("bullets") or []):
            pointed_at.append((entry_number, bullet_number))
    return pointed_at


def _skill_lines_it_points_at(pointers, skill_lines: int) -> list:
    """The reply's pointers at skills lines, as line numbers; one the résumé lacks is dropped."""
    pointed_at = []
    for pointer in pointers or []:
        written = str(pointer).strip()
        if written.isdigit() and 1 <= int(written) <= skill_lines:
            pointed_at.append(int(written))
    return pointed_at


def _one_keyword_each(requirement: str) -> list:
    """A requirement naming several keywords as one ("Python, Pandas, and FastAPI"), taken apart into one each.

    Otherwise a bullet naming Python would count as showing Pandas too.
    """
    parts = [part.strip(" .:") for part in _KEYWORD_PARTS.split(requirement)]
    parts = [part for part in parts if part]
    # Only a list of names is taken apart; "product managers, architects and QA teams" is one duty.
    is_a_list_of_names = len(parts) >= 2 and all(
        len(part.split()) <= MOST_WORDS_IN_A_KEYWORD and part != part.lower() for part in parts)
    return parts if is_a_list_of_names else [requirement]
