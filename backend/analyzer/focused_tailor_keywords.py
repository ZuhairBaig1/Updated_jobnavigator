"""The posting's keywords in a focused or complete copy: written as the posting spells them, and counted (step 6)."""
import re

from backend.analyzer import resume_schema as rs
from backend.analyzer.focused_tailor_basics import DUTY, MOST_WORDS_IN_A_KEYWORD
from backend.analyzer.focused_tailor_checks import _DETAIL, _KEYWORD_PARTS, _however_it_is_spelled, _is_stated


def keywords_of_the_posting(requirements: list, jd_text: str, base: dict | None = None) -> list:
    """The posting's keywords, each exactly as the posting spells it.

    They are every skill the résumé lists that the posting also names, however the two spell it ("GitHub" where
    the posting writes "Github"), and the skills and qualifications the reply found in the posting. Only a name is
    kept, not an ordinary lower-case word.
    """
    found = []
    for keyword in _postings_spelling_of_the_resumes_skills(base, jd_text):
        if keyword not in found:
            found.append(keyword)
    for keyword in _keywords_of_the_requirements(requirements, jd_text):
        if keyword not in found:
            found.append(keyword)
    return found


def _postings_spelling_of_the_resumes_skills(base: dict | None, jd_text: str) -> list:
    """Each skill the résumé lists that the posting names, as the posting writes it."""
    spellings = []
    for value in ((base or {}).get("skills") or {}).values():
        for term in rs._split_skills(value):
            name = _DETAIL.sub("", term).strip()
            if len(rs._squash(name)) < 3:
                continue
            written = _however_it_is_spelled(name).search(jd_text or "")
            if written and _has_a_capital(written.group(0)):
                spellings.append(written.group(0))
    return spellings


def _keywords_of_the_requirements(requirements: list, jd_text: str) -> list:
    """The keywords of the posting's skills and qualifications (not its duties) that the posting writes letter for letter."""
    keywords = []
    for requirement in requirements:
        if requirement["importance"] == DUTY:
            continue
        # A requirement may hold several keywords: "CI / CD (Github Actions)".
        for part in _KEYWORD_PARTS.split(requirement["requirement"]):
            keyword = part.strip(" .:")
            if (keyword and keyword in jd_text and len(rs._squash(keyword)) >= 3 and _has_a_capital(keyword)
                    and len(keyword.split()) <= MOST_WORDS_IN_A_KEYWORD):
                keywords.append(keyword)
    return keywords


def _has_a_capital(text: str) -> bool:
    """Whether the text has a capital letter, so it reads as a name and not an ordinary word."""
    return text != text.lower()


def in_the_postings_spelling(text: str, keywords: list) -> str:
    """The text with each keyword it already holds respelled as the posting spells it: "CSS3" as "Css3", "CI/CD" as "CI / CD".

    Only the same word in another spelling is touched, so nothing is claimed that was not claimed before.
    Longer keywords go first, so "Test Driven Development" is settled before "Development" could be.
    """
    for keyword in sorted(keywords, key=len, reverse=True):
        text = _however_it_is_spelled(keyword).sub(lambda match: keyword, text or "")
    return text


def _respell_the_keywords(copy: dict, keywords: list) -> None:
    """Writes every keyword the copy holds the way the posting spells it, in the summary, the skills and the bullets."""
    copy["summary"] = in_the_postings_spelling(copy["summary"], keywords)
    copy["skills"] = {label: in_the_postings_spelling(value, keywords) for label, value in copy["skills"].items()}
    for entry in copy["experience"] + copy["projects"]:
        entry["bullets"] = [in_the_postings_spelling(bullet, keywords) for bullet in entry["bullets"]]


def with_the_postings_name_beside_the_resumes(skills: dict, other_names: dict) -> dict:
    """The skills, with the posting's name for a skill added in brackets where only the résumé's name is printed.

    "SQL Server" becomes "SQL Server (MSSQL)", so the posting's keyword is on the page and the résumé's own name is not lost.
    """
    everything = " | ".join(str(value) for value in skills.values())
    for posting_name, resume_name in other_names.items():
        if _is_stated(posting_name, everything):
            continue
        for label, value in skills.items():
            terms = rs._split_skills(value)
            position = _position_of_the_skill(resume_name, terms)
            if position is not None:
                terms[position] = f"{terms[position]} ({posting_name})"
                skills = {**skills, label: ", ".join(terms)}
                break
    return skills


def _position_of_the_skill(name: str, terms: list) -> int | None:
    for position, term in enumerate(terms):
        if rs._squash(_DETAIL.sub("", term)) == rs._squash(name):
            return position
    return None


def keywords_counted(copy: dict, keywords: list) -> list:
    """For each keyword of the posting: whether the copy's skills name it exactly, and how often the rest of the copy does."""
    skills = " | ".join(str(value) for value in (copy.get("skills") or {}).values())
    rest_of_the_copy = [str(copy.get("summary") or "")]
    for entry in (copy.get("experience") or []) + (copy.get("projects") or []):
        rest_of_the_copy += entry.get("bullets") or []
    elsewhere = " ".join(rest_of_the_copy)
    counted = []
    for keyword in keywords:
        exactly = _exactly(keyword)
        counted.append({"keyword": keyword, "in_skills": bool(exactly.search(skills)),
                        "elsewhere": len(exactly.findall(elsewhere))})
    return counted


def _exactly(keyword: str):
    """A pattern for the keyword exactly as written, as a word of its own."""
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(keyword) + r"(?![A-Za-z0-9+#])")
