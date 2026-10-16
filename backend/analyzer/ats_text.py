"""The résumé and the posting as the ATS scorers read them: redacted, numbered, and without lines aimed at the scorer.

Shared by the Default and Fast (Jev) ATS scorers, so both read the same text. What it does, and why
(see the ATS scoring rules, sections 6-8):
  - The model never sees the candidate's name, contact details, employers, locations, schools or years of
    graduation. None bears on the match, and two studies found LLM scores moved with names and schools.
  - Roles, bullets, education and certification lines are numbered so a model can cite them, and code can
    check that what it cites exists.
  - The posting's lines are numbered too, so every requirement points at the line it came from.
  - A line written to whoever scores the résumé ("rate this candidate…") is removed before any model reads it.
"""
import re
from datetime import date

from backend.analyzer import line_match
from backend.analyzer import rubric_ats_score as rubric
from backend.analyzer.job_requirements import clean_job_text

RESUME_MAX_CHARS = rubric.RESUME_MAX_CHARS
POSTING_MAX_CHARS = rubric.POSTING_MAX_CHARS

A_ROLE_LINE = re.compile(r"^(?P<title>.*?) at (?P<company>.*?) \((?P<rest>.*)$")
AN_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
A_LINK = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
A_PHONE_NUMBER = re.compile(r"\+?\d[\d\s().-]{7,}\d")
A_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
SECTIONS_KEPT = ("summary", "experience", "projects", "education", "certifications", "publications")
# A line written to whoever, or whatever, scores the résumé, rather than about the candidate's work.
SPEAKS_TO_THE_EVALUATOR = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) (?:instructions|prompts?)"
    r"|note to (?:the )?(?:evaluator|reviewer|recruiter|screener|ai|model|assistant)"
    r"|(?:dear|attention|hey) (?:ai|assistant|chatgpt|gpt|claude|model|evaluator|ats)\b"
    r"|\b(?:rate|score|rank|grade)\b[^.]{0,40}\bthis (?:candidate|resume|résumé|applicant)\b"
    r"|\bthis (?:candidate|applicant)\b[^.]{0,40}\b(?:meets|matches|satisfies|fulfils|fulfills) (?:every|all)\b"
    r"|\b(?:rate|score|mark) (?:it|me|this|them) (?:100|10/10|ten|the highest)\b",
    re.IGNORECASE)

# A model may write an id with its text after it ("[J10] Write clean code"); only the id is read.
A_LINE_ID = re.compile(r"J\d+")
A_BULLET_ID = re.compile(r"P?\d+\.\d+")
A_RESUME_ID = re.compile(r"P?\d+\.\d+|E\d+|C\d+")


def without_lines_to_the_evaluator(text: str) -> tuple:
    """The text without lines that speak to the evaluator, and the lines removed."""
    kept, removed = [], []
    for line in (text or "").splitlines():
        (removed if SPEAKS_TO_THE_EVALUATOR.search(line) else kept).append(line)
    return "\n".join(kept), removed


def first_id(value, pattern) -> str:
    found = pattern.search(str(value or ""))
    return found.group(0) if found else ""


# ── The résumé's sections and roles ────────────────────────────────────────────

def sections(resume_text: str) -> list:
    """[(heading in lower case, its lines)] for every "## " section, in order."""
    found, heading, lines = [], None, []
    for line in (resume_text or "").splitlines():
        if line.startswith("## "):
            if heading is not None:
                found.append((heading, lines))
            heading, lines = line[3:].strip().lower(), []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        found.append((heading, lines))
    return found


def section_lines(resume_text: str, name: str) -> list:
    return [line for heading, lines in sections(resume_text) if heading == name for line in lines]


def roles_in(resume_text: str, today: date | None = None) -> list:
    """The roles of the Experience section in order: [{"title", "months", "bullets"}].

    months is the set of months the role's dates cover (empty when it states none).
    """
    roles = []
    for line in section_lines(resume_text, "experience"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            if roles:
                roles[-1]["bullets"].append(stripped.lstrip("-•* ").strip())
            continue
        found = A_ROLE_LINE.match(stripped)
        is_a_role = found or rubric.DATES_OF_A_ROLE.search(stripped)
        if is_a_role or not roles:
            title = found.group("title").strip() if found else stripped.split("(")[0].strip()
            roles.append({"title": title, "months": _months_of(stripped, today), "bullets": []})
        else:
            roles[-1]["bullets"].append(stripped)   # a role's description line counts as one of its bullets
    return roles


def _months_of(line: str, today: date | None = None) -> set:
    today = today or date.today()
    months = set()
    for start_month, start_year, still_there, end_month, end_year in rubric.DATES_OF_A_ROLE.findall(line):
        first = rubric._month_number(start_year, start_month, 1)
        last = (rubric._month_number(today.year, rubric.MONTHS[today.month - 1], 12) if still_there
                else rubric._month_number(end_year, end_month, 12))
        months.update(range(first, last + 1))
    return months


# ── The résumé as a model reads it ─────────────────────────────────────────────

def redacted_resume(resume_text: str, title_lines=(), today: date | None = None) -> str:
    """The résumé with nothing that identifies the person: no name, contact details, employers, locations,
    schools or years of graduation. Roles and bullets are numbered so a model can cite them; the skills
    list is left out, so a model judges the work rather than the list."""
    name_words = _words_of_the_name(resume_text)
    parts = []
    if title_lines:
        parts += ["## Title"] + [str(line) for line in title_lines if str(line).strip()]
    for heading, lines in sections(resume_text):
        if heading not in SECTIONS_KEPT:
            continue
        parts.append(f"## {heading.capitalize()}")
        if heading == "experience":
            for number, role in enumerate(roles_in(resume_text, today), start=1):
                parts.append(f"Role {number}: {role['title']}")
                parts += [f"[{number}.{bullet}] {text}" for bullet, text in enumerate(role["bullets"], start=1)]
        elif heading == "projects":
            parts += _numbered_projects(lines)
        elif heading == "education":
            parts += [line.split(" — ")[0].strip() for line in lines if line.strip()]
        else:
            parts += [line for line in lines if line.strip()]
    text = "\n".join(parts)
    text = A_LINK.sub("", AN_EMAIL.sub("", text))
    text = A_PHONE_NUMBER.sub(lambda found: found.group(0) if looks_like_a_range(found.group(0)) else "", text)
    for word in name_words:
        text = re.sub(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", "the candidate", text)
    text = _without_graduation_years(text)
    return text.strip()[:RESUME_MAX_CHARS]


def resume_for_the_model(resume_text: str, title_lines=(), today: date | None = None) -> str:
    """The redacted résumé, with each role's length instead of its dates, and the education and certification
    lines numbered so they can be cited."""
    redacted = redacted_resume(resume_text, title_lines, today)
    roles = roles_in(resume_text, today)
    out, section, numbers = [], "", {"education": 0, "certifications": 0}
    for line in redacted.splitlines():
        if line.startswith("## "):
            section = line[3:].strip().lower()
            out.append(line)
            continue
        found = re.match(r"^Role (\d+): (.*)$", line)
        if found and 1 <= int(found.group(1)) <= len(roles):
            months = len(roles[int(found.group(1)) - 1]["months"])
            line = f"{line} ({round(months / 12, 1):g} years)" if months else line
        elif section in numbers and line.strip():
            numbers[section] += 1
            line = f"[{'E' if section == 'education' else 'C'}{numbers[section]}] {line.lstrip('- ')}"
        out.append(line)
    return "\n".join(out)


def citable_lines(model_text: str) -> dict:
    """{id: the text of that bullet or line} for everything a model can cite."""
    return dict(re.findall(r"^\[((?:P?\d+\.\d+)|E\d+|C\d+)\] (.*)$", model_text, re.MULTILINE))


def _words_of_the_name(resume_text: str) -> list:
    """The name is the first line of the scoring text, before any section heading."""
    first = next((line.strip() for line in (resume_text or "").splitlines() if line.strip()), "")
    if first.startswith("## "):
        return []
    return [word for word in re.findall(r"[A-Za-z][A-Za-z'-]+", first) if len(word) > 2]


def looks_like_a_range(text: str) -> bool:
    """"2019 - 2022" or "8 - 10" is not a phone number."""
    return len(re.sub(r"\D", "", text)) < 9


def _without_graduation_years(text: str) -> str:
    """Years are removed from the Education and Certifications lines only; the roles keep no dates anyway."""
    out, in_a_dated_section = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            in_a_dated_section = line[3:].strip().lower() in ("education", "certifications")
        elif in_a_dated_section:
            line = re.sub(r"\s{2,}", " ", A_YEAR.sub("", line)).strip(" ,;–—-")
        out.append(line)
    return "\n".join(out)


def _numbered_projects(lines: list) -> list:
    out, project, bullet = [], 0, 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            bullet += 1
            out.append(f"[P{max(project, 1)}.{bullet}] {stripped.lstrip('-•* ').strip()}")
        else:
            project, bullet = project + 1, 0
            out.append(f"Project {project}: {stripped}")
    return out


# ── The posting ────────────────────────────────────────────────────────────────

def numbered_posting(job_text: str) -> tuple:
    """The posting as numbered lines for the model, and {id: line}."""
    lines = line_match.posting_lines(clean_job_text(line_match.posting_as_plain_text(job_text)))
    by_id = {f"J{number}": line for number, line in enumerate(lines, start=1)}
    text = "\n".join(f"[{line_id}] {line}" for line_id, line in by_id.items())
    return text[:POSTING_MAX_CHARS], by_id


def length_warnings(resume_text: str, job_text: str) -> list:
    """What will be cut before a model reads it, said up front rather than discovered in the score."""
    warnings = []
    if len(resume_text or "") > RESUME_MAX_CHARS:
        warnings.append(f"The résumé is {len(resume_text):,} characters; only the first {RESUME_MAX_CHARS:,} are read.")
    if len(job_text or "") > POSTING_MAX_CHARS:
        warnings.append(f"The posting is {len(job_text):,} characters; only the first {POSTING_MAX_CHARS:,} are read.")
    return warnings
