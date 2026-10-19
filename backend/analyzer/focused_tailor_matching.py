"""Step 1 of a focused or complete copy: reading the reply's matching of the posting to the résumé.

The reply lists what the posting asks for, how much it wants each thing, and where the résumé shows it. This keeps
only what can be checked: a pointer must lead to a bullet or a skills line the résumé has, and the posting's name
for something the résumé names differently must truly be the same thing.
"""
import re

from backend.analyzer import resume_schema as rs
from backend.analyzer.focused_tailor_basics import FROM_A_JOB, FROM_A_PROJECT, FROM_THE_SKILLS, MOST_WORDS_IN_A_KEYWORD
from backend.analyzer.focused_tailor_checks import _KEYWORD_PARTS, _adds_words_to, _is_stated

# Where a pointer leads inside its source: "2.5" is bullet 5 of job (or project) 2, "4" is skills line 4.
_PLACE_NUMBERS = re.compile(r"^\s*(\d+)(?:\s*\.\s*(\d+))?\s*$")
# An older reply wrote the source into a text pointer ("job 1.5"), or left it out ("1.5" is a job's bullet).
_SOURCE_WORD = re.compile(r"^\s*(job|project|skills)\s+", re.IGNORECASE)

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

    The reply writes each as {"name", "kind", "has", "by": [{"source": "job", "at": "1.5"}, ...]}; each comes back as
    {"requirement", "importance", "shown_by", "resume_has_it", "found_at"}.
    `found_at` holds every place that exists, by kind: {"job": [(job, bullet)], "project": [(project, bullet)],
    "skills": [line]}. `shown_by` is its job places, since only a job's bullets are chosen by what they show.
    """
    matched = []
    for entry in (reply or {}).get("requirements") or []:
        if not isinstance(entry, dict) or not rs._text(entry.get("name")):
            continue
        found_at = _places_it_points_at(entry.get("by"), roles, projects, skill_lines)
        resume_has_it = bool(entry.get("has")) or any(found_at.values())
        for name in _one_keyword_each(rs._text(entry["name"])):
            matched.append({"requirement": name, "importance": entry.get("kind"), "shown_by": found_at[FROM_A_JOB],
                            "resume_has_it": resume_has_it, "found_at": found_at})
    return matched


def _places_it_points_at(pointers, roles: list, projects: list, skill_lines: int) -> dict:
    """The reply's pointers that lead to a place the résumé has, by kind; any other is dropped."""
    found_at = {FROM_A_JOB: [], FROM_A_PROJECT: [], FROM_THE_SKILLS: []}
    for pointer in pointers or []:
        place = _place(pointer, roles, projects, skill_lines)
        if place:
            kind, where = place
            found_at[kind].append(where)
    return found_at


def _place(pointer, roles: list, projects: list, skill_lines: int):
    """One pointer as (source, where it leads) when the résumé has that place; None otherwise."""
    source, place = _source_and_place(pointer)
    numbers = _PLACE_NUMBERS.match(place)
    if not numbers or source not in (FROM_A_JOB, FROM_A_PROJECT, FROM_THE_SKILLS):
        return None
    number, bullet = int(numbers.group(1)), numbers.group(2)
    if source == FROM_THE_SKILLS:
        return (source, number) if bullet is None and 1 <= number <= skill_lines else None
    entries = roles if source == FROM_A_JOB else projects
    if bullet is None or not 1 <= number <= len(entries):
        return None
    bullet = int(bullet)
    if 1 <= bullet <= len(entries[number - 1].get("bullets") or []):
        return (source, (number, bullet))
    return None


def _source_and_place(pointer) -> tuple:
    """A pointer's source and its place in that source: {"source": "job", "at": "1.5"} is ("job", "1.5")."""
    if isinstance(pointer, dict):
        return str(pointer.get("source") or "").lower(), str(pointer.get("at") or "")
    text = str(pointer)
    written = _SOURCE_WORD.match(text)
    if written:
        return written.group(1).lower(), text[written.end():]
    return FROM_A_JOB, text


def requirements_met_in_the_copy(reply: dict) -> list:
    """The reply's own check of the copy it wrote, one entry a requirement: {"requirement", "met", "found_at"}.

    Its places point into what the reply itself wrote: a role's or project's bullets by their position in the reply,
    and its skills categories by their position. A place the reply did not write is dropped.
    """
    written_bullets = {FROM_A_JOB: _bullets_written(reply.get("work_experience"), "role_number"),
                       FROM_A_PROJECT: _bullets_written(reply.get("projects"), "project_number")}
    skill_groups = len(reply.get("technical_skills") or [])
    met = []
    for entry in (reply or {}).get("requirements_met") or []:
        if not isinstance(entry, dict) or not rs._text(entry.get("name")):
            continue
        found_at = {FROM_A_JOB: [], FROM_A_PROJECT: [], FROM_THE_SKILLS: []}
        for pointer in entry.get("by") or []:
            place = _place_in_what_was_written(pointer, written_bullets, skill_groups)
            if place:
                source, where = place
                found_at[source].append(where)
        met.append({"requirement": rs._text(entry["name"]), "met": bool(entry.get("met")), "found_at": found_at})
    return met


def _bullets_written(entries, number_key: str) -> dict:
    """How many bullets the reply wrote for each role or project number; the first mention of a number wins."""
    counts = {}
    for entry in entries or []:
        if isinstance(entry, dict) and isinstance(entry.get(number_key), int) and entry[number_key] not in counts:
            counts[entry[number_key]] = len(entry.get("bullets") or [])
    return counts


def _place_in_what_was_written(pointer, written_bullets: dict, skill_groups: int):
    """One pointer into the reply's own writing as (source, where it leads), or None when the reply wrote no such place."""
    source, place = _source_and_place(pointer)
    numbers = _PLACE_NUMBERS.match(place)
    if not numbers or source not in (FROM_A_JOB, FROM_A_PROJECT, FROM_THE_SKILLS):
        return None
    number, bullet = int(numbers.group(1)), numbers.group(2)
    if source == FROM_THE_SKILLS:
        return (source, number) if bullet is None and 1 <= number <= skill_groups else None
    if bullet is None or not 1 <= int(bullet) <= written_bullets[source].get(number, 0):
        return None
    return (source, (number, int(bullet)))


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
