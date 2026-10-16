"""The posting's search terms, matched in the résumé the way a recruiter's ATS search matches them.

A model picks the terms (ATS scoring rules 10.3: the model is the backbone); this code only checks them:
  - a term must appear in the posting line it came from, or it is a concept, not a search term (rule 14);
  - a term is found in the résumé literally, allowing only the safe differences a search box or a reader
    treats as the same word (10.4): letter case, spacing and punctuation ("CI/CD", "CI CD"), plurals,
    "RESTful" for "REST", and the skills dictionary's other spellings of the same tool ("K8s" for
    Kubernetes, "NodeJS" for Node.js, "Postgres" for PostgreSQL);
  - a narrower name never counts for a broader one: "PostgreSQL" is not a match for "SQL";
  - the same term asked for twice counts once, at the weight of its most important mention.
"""
import re

from backend.analyzer import ats_text, keyword_match

# Weight of a requirement's kind, shared by every version that uses these terms.
KIND_WEIGHT = {"must": 3, "duty": 2, "nice": 1}
# A term the résumé does not write, shown under another name that was checked, earns half (ATS rules A4.10).
OTHER_NAME_POINTS = 0.5
FOUND, OTHER_NAME, NOT_FOUND = "found", "other_name", "not_found"
SAME_WORD = {"restful": "rest"}
A_WORD = re.compile(r"[a-z0-9+#]+")


# ── Folding: the safe differences ──────────────────────────────────────────────

def folded_words(text: str) -> list:
    """The words of the text in the form two spellings of the same term share."""
    return [_folded(word) for word in A_WORD.findall((text or "").lower())]


def _folded(word: str) -> str:
    word = SAME_WORD.get(word, word)
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _contains(words: list, phrase: list) -> bool:
    if not phrase:
        return False
    size = len(phrase)
    return any(words[start:start + size] == phrase for start in range(len(words) - size + 1))


def standard_name(term: str) -> str | None:
    """The skills dictionary's standard name for the term, or None when it does not know it."""
    words = (term or "").split()
    if not words:
        return None
    entry = keyword_match._lookup(words)
    if entry and keyword_match._is_written_as_a_skill(words[0], needs_capital=entry[1]):
        return entry[0]
    return None


def appears_in(term: str, text: str, names_in_text: dict | None = None) -> bool:
    """Whether the text writes the term, allowing only the safe differences."""
    if _contains(folded_words(text), folded_words(term)):
        return True
    name = standard_name(term)
    if name is None:
        return False
    names = names_in_text if names_in_text is not None else keyword_match.skills_in(text)
    return name in names


# ── Terms of a requirement ─────────────────────────────────────────────────────

def grounded(terms: list, posting_line: str) -> tuple:
    """(the terms the posting line really contains, the terms dropped because it does not)."""
    kept, dropped = [], []
    for term in terms or []:
        term = str(term or "").strip()
        if not term:
            continue
        (kept if appears_in(term, posting_line) else dropped).append(term)
    return kept, dropped


def same_term_key(term: str) -> str:
    """Two terms with the same key are the same search term."""
    return standard_name(term) or " ".join(folded_words(term))


def where_in_the_resume(term: str, resume_text: str) -> str:
    """"in a role" when a role or project writes it, "listed" when only the summary, skills or other lines do."""
    worked ="\n".join(line for heading, lines in ats_text.sections(resume_text)
                       if heading in ("experience", "projects") for line in lines)
    return "in a role" if appears_in(term, worked) else "listed"


def keyword_part(requirements: list, resume_text: str) -> dict:
    """The share of the posting's search terms the résumé writes, weighted by the kind of requirement they
    come from. Each requirement is {"kind", "terms", "wording"}; wording "other" (another name, checked)
    earns half for its terms. None when the posting names no search terms."""
    names_in_resume = keyword_match.skills_in(resume_text)
    terms = {}
    for item in requirements:
        for term in item.get("terms") or []:
            key = same_term_key(term)
            weight = KIND_WEIGHT.get(item.get("kind"), 1)
            if key not in terms or weight > terms[key]["weight"]:
                terms[key] = {"term": term, "kind": item.get("kind"), "weight": weight,
                              "other_name": item.get("wording") == "other"}
    rows = []
    for entry in terms.values():
        if appears_in(entry["term"], resume_text, names_in_resume):
            status, points = FOUND, 1.0
        elif entry["other_name"]:
            status, points = OTHER_NAME, OTHER_NAME_POINTS
        else:
            status, points = NOT_FOUND, 0.0
        rows.append({"term": entry["term"], "kind": entry["kind"], "status": status, "points": points,
                     "where": where_in_the_resume(entry["term"], resume_text) if status == FOUND else ""})
    weights = sum(KIND_WEIGHT.get(row["kind"], 1) for row in rows)
    share = (sum(KIND_WEIGHT.get(row["kind"], 1) * row["points"] for row in rows) / weights) if rows else None
    return {"share": None if share is None else round(share, 3), "terms": rows,
            "found": [row["term"] for row in rows if row["status"] == FOUND],
            "other_name": [row["term"] for row in rows if row["status"] == OTHER_NAME],
            "missing": [row["term"] for row in rows if row["status"] == NOT_FOUND]}


def wording_of(terms: list, resume_text: str, other_name_checked: bool) -> str | None:
    """How the résumé words a requirement: "exact" when it writes one of the posting's terms, "other" when
    another name for it was checked, "none" otherwise; None when the requirement has no search terms."""
    if not terms:
        return None
    if any(appears_in(term, resume_text) for term in terms):
        return "exact"
    return "other" if other_name_checked else "none"
