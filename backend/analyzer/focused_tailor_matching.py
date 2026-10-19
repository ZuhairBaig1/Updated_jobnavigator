"""Step 1 of a focused or complete copy: reading the reply's matching of the posting to the résumé.

The reply lists what the posting asks for, how much it wants each thing, and where the résumé shows it. This keeps
only what can be checked: a pointer must lead to a bullet or a skills line the résumé has, and the posting's name
for something the résumé names differently must truly be the same thing.
"""
import re

from backend.analyzer import resume_schema as rs
from backend.analyzer.focused_tailor_basics import (DUTY, FROM_A_JOB, FROM_A_PROJECT, FROM_THE_SKILLS,
                                                    MOST_WORDS_IN_A_KEYWORD, MUST_HAVE, NICE_TO_HAVE, _projects)
from backend.analyzer.focused_tailor_checks import _KEYWORD_PARTS, _adds_words_to, _is_stated

# Where a pointer leads inside its source: "2.5" is bullet 5 of job (or project) 2, "4" is skills line 4.
_PLACE_NUMBERS = re.compile(r"^\s*(\d+)(?:\s*\.\s*(\d+))?\s*$")
# An older reply wrote the source into a text pointer ("job 1.5"), or left it out ("1.5" is a job's bullet).
_SOURCE_WORD = re.compile(r"^\s*(job|project|skills)\s+", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9+#]+")
# The share of its words a bullet of the reply keeps once the app's checks have made it the copy's bullet.
SAME_BULLET = 0.6

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


# ── What a Tailor + Jev ATS score reads from the copy (ats_score_tailor_jev.py) ─

def requirements_listed(reply: dict) -> list:
    """The posting's requirements as the reply listed them, [{"name", "kind"}]."""
    listed = []
    for entry in (reply or {}).get("requirements") or []:
        if isinstance(entry, dict) and rs._text(entry.get("name")) and entry.get("kind") in (MUST_HAVE, NICE_TO_HAVE, DUTY):
            listed.append({"name": rs._text(entry["name"]), "kind": entry["kind"]})
    return listed


def with_the_lines_of_the_copy(met: list, reply: dict, base: dict, copy_data: dict) -> list:
    """Each entry of requirements_met with "shown_in": [{"where", "text"}], the copy's own lines at the places it points
    to. The places point into the reply, so each is found again among the copy's lines after the app's checks; a
    bullet the checks rewrote or swapped out is not found, and is left out."""
    for entry in met:
        lines = []
        for source, places in entry["found_at"].items():
            for where in places:
                line = _line_of_the_copy(source, where, reply, base, copy_data)
                if line and line not in lines:
                    lines.append(line)
        entry["shown_in"] = lines
    return met


def _line_of_the_copy(source: str, where, reply: dict, base: dict, copy_data: dict) -> dict | None:
    if source == FROM_THE_SKILLS:
        groups = [group for group in reply.get("technical_skills") or [] if isinstance(group, dict)]
        category = rs._text(groups[where - 1].get("category")) if where <= len(groups) else ""
        skills = (copy_data.get("skills") or {}).get(category) if category else None
        return {"where": "Skills", "text": f"{category}: {skills}"} if skills else None
    number, bullet = where
    if source == FROM_A_JOB:
        written = _bullet_written(reply.get("work_experience"), "role_number", number, bullet)
        roles = copy_data.get("experience") or []
        kept = roles[number - 1] if number <= len(roles) else {}
        label = rs._text(kept.get("title")) or f"Role {number}"
    else:
        # A focused copy can drop projects, so a project is found again by its name.
        written = _bullet_written(reply.get("projects"), "project_number", number, bullet)
        projects = _projects(base)
        name = rs._text(projects[number - 1].get("name")) if number <= len(projects) else ""
        kept = next((project for project in copy_data.get("projects") or [] if rs._text(project.get("name")) == name), {})
        label = name or "Project"
    best = max(kept.get("bullets") or [], key=lambda text: _words_shared(written, text), default="")
    return {"where": label, "text": best} if best and _words_shared(written, best) >= SAME_BULLET else None


def _bullet_written(entries, number_key: str, number: int, bullet: int) -> str:
    """The text the reply wrote for that bullet of that role or project; the first mention of a number wins."""
    entry = next((e for e in entries or [] if isinstance(e, dict) and e.get(number_key) == number), {})
    bullets = entry.get("bullets") or []
    if not 1 <= bullet <= len(bullets):
        return ""
    written = bullets[bullet - 1]
    return rs._text(written.get("text") if isinstance(written, dict) else written)


def _words_shared(one: str, other: str) -> float:
    """The share of their words two sentences have in common."""
    first, second = set(_WORD.findall(str(one).lower())), set(_WORD.findall(str(other).lower()))
    return len(first & second) / len(first | second) if first | second else 0.0


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
