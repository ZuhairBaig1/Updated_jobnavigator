"""What a Tailor + Jev score found, said in plain words, and where the copy shows each requirement.

Jev answers every question with a pick and how sure it is; it never writes why. So code writes the issues, each
from an answer already in the score and from what code itself knows (which of the posting's words the résumé
writes, how long each role lasted, the posting's title), and the issues never disagree with the score. They say
what falls short, not why Jev judged so.

"Shown in" comes from two places: the tailoring's own check of the copy it wrote (requirements_met, each place
found again among the copy's lines when it was tailored) and the lines that write the requirement's search term.

Tested 2026-10-19 on 8 focused copies: 89% of the requirements Jev judged shown got a line, nearly all from the
tailoring, and about 9 in 10 of those looked right by hand. The format checks are not repeated here: they have
their own table beside the score.
"""
from backend.analyzer import ats_score_default, ats_text, posting_keywords
from backend.analyzer import resume_schema as rs

MOST_LINES_SHOWN = 3
LOW_CONFIDENCE = 0.5
WEAK = ("none", "listed", "partly")
SAID_AS = {
    "none": "not in your résumé.",
    "listed": "only listed (skills, summary or education); no role or project shows you using it.",
    "partly": "a role or project shows only part of it, or something close to it.",
    "in a role": "named in a role, but no bullet shows you doing it.",
    "unconfirmed": "Jev found nothing, though the tailoring points to the line below; check it.",
}


# ── Shown in ───────────────────────────────────────────────────────────────────

def shown_in(requirement: dict, requirements_met: list, resume_text: str) -> list:
    """Up to MOST_LINES_SHOWN lines of the copy that show the requirement, [{"where", "text", "found_by"}]: the
    tailoring's places first, then the lines writing its search term. A line the copy no longer has (it was edited
    after tailoring) is left out."""
    met = next((entry for entry in requirements_met or [] if entry.get("met")
                and rs._squash(entry.get("requirement") or "") == rs._squash(requirement["requirement"])), {})
    lines = [{**line, "found_by": "tailoring"} for line in met.get("shown_in") or []
             if line.get("text") and line["text"] in resume_text]
    for term in requirement.get("terms") or []:
        lines += [{**line, "found_by": "search term"} for line in lines_writing(term, resume_text)]
    unique = []
    for line in lines:
        if all(line["text"] != kept["text"] for kept in unique):
            unique.append(line)
    return unique[:MOST_LINES_SHOWN]


def lines_writing(term: str, resume_text: str) -> list:
    """The résumé's lines that write the term: each role's bullets under the role's title, other lines under their section."""
    found = []
    for role in ats_text.roles_in(resume_text):
        found += [{"where": role["title"], "text": bullet} for bullet in role["bullets"]
                  if posting_keywords.appears_in(term, bullet)]
    for heading, lines in ats_text.sections(resume_text):
        if heading != "experience":
            found += [{"where": heading.capitalize(), "text": line.strip().lstrip("-•* ").strip()} for line in lines
                      if line.strip() and posting_keywords.appears_in(term, line)]
    return found


def where_the_terms_are(terms: list, resume_text: str) -> str | None:
    """"in a role" when a role or project writes one of the terms, "listed" when only other lines do, "nowhere"
    when none is written; None for a requirement without terms."""
    if not terms:
        return None
    written = [term for term in terms if posting_keywords.appears_in(term, resume_text)]
    if not written:
        return "nowhere"
    if any(posting_keywords.where_in_the_resume(term, resume_text) == "in a role" for term in written):
        return "in a role"
    return "listed"


# ── The issues ─────────────────────────────────────────────────────────────────

def issues_of(requirements: list, keywords: dict, seniority: dict, title: dict, occupation: dict, left_out: dict,
              resume_text: str, today=None) -> list:
    """[{"about", "issue", "shown_in"}], the most important first; each said only from an answer in the score."""
    issues = [_must_have_issue(item, resume_text) for item in sorted(requirements, key=lambda item: item["points"])
              if item["kind"] == "must" and item["evidence"] in WEAK]
    missing_nice = [item["requirement"] for item in requirements if item["kind"] == "nice" and item["evidence"] == "none"
                    and not item.get("shown_in") and where_the_terms_are(item.get("terms"), resume_text) in (None, "nowhere")]
    if missing_nice:
        issues.append(_issue("nice-to-have", "Not in your résumé: " + ", ".join(missing_nice) + "."))
    issues += _keyword_issues(keywords)
    if seniority.get("share") is not None and seniority["share"] < 1:
        asked = "; ".join(left_out.get("years") or []) or "the years or level the posting asks for"
        issues.append(_issue("experience", f"The posting asks for {asked}; your roles add up to about "
                             f"{years_of_all_roles(resume_text, today):g} years. Jev: "
                             f"{ats_score_default.SENIORITY_MEANING[seniority['category']]} "
                             f"({_how_sure(seniority, seniority['category'])})."))
    if title.get("share", 1) < 1:
        issues.append(_issue("title", f"Your résumé does not write the posting's title \"{title.get('posting_title', '')}\". "
                             f"Jev: {ats_score_default.TITLE_MEANING[title['category']]} ({_how_sure_of_the_title(title)})."))
    if occupation.get("category") != "same":
        issues.append(_issue("occupation", f"Jev judged your roles {ats_score_default.OCCUPATION_MEANING[occupation['category']]} "
                             f"to this job ({_how_sure(occupation, occupation['category'])})."))
    unsure = [item["requirement"] for item in requirements
              if item.get("confidence") is not None and item["confidence"] < LOW_CONFIDENCE]
    if unsure:
        issues.append(_issue("unsure", "Jev was not sure about: " + ", ".join(unsure) + ". Check these yourself."))
    not_scored = [f"personal traits ({', '.join(left_out['traits'])})" if left_out.get("traits") else "",
                  "years of experience (seniority covers them)" if left_out.get("years") else ""]
    if any(not_scored):
        issues.append(_issue("not scored", "Not scored: " + "; ".join(part for part in not_scored if part) + "."))
    return issues


def _must_have_issue(item: dict, resume_text: str) -> dict:
    """What falls short of a weak must-have, from Jev's evidence level, with code's word check where it is sure: a
    search term the résumé writes is never "not in your résumé", and nor is a requirement the tailoring found a
    line for (then Jev and the tailoring disagree, and the line is shown to be checked)."""
    where = where_the_terms_are(item.get("terms"), resume_text)
    if item["evidence"] == "partly":
        falls_short = "partly"
    elif item["evidence"] == "listed" or where == "listed":
        falls_short = "listed"
    elif where == "in a role":
        falls_short = "in a role"
    elif item.get("shown_in"):
        falls_short = "unconfirmed"
    else:
        falls_short = "none"
    shown = item.get("shown_in")[:1] if falls_short != "none" and item.get("shown_in") else []
    return _issue("must-have", f"{item['requirement']}: {SAID_AS[falls_short]}", shown)


def _keyword_issues(keywords: dict) -> list:
    rows = keywords.get("terms") or []
    missing = sorted((row for row in rows if row["status"] == posting_keywords.NOT_FOUND),
                     key=lambda row: -posting_keywords.KIND_WEIGHT.get(row["kind"], 1))
    other_name = [row["term"] for row in rows if row["status"] == posting_keywords.OTHER_NAME]
    issues = []
    if missing:
        issues.append(_issue("keywords", "An ATS search for these words finds nothing in your résumé: "
                             + ", ".join(row["term"] for row in missing) + "."))
    if other_name:
        issues.append(_issue("keywords", "Shown under another name, so a search for the posting's word misses it: "
                             + ", ".join(other_name) + "."))
    return issues


def _issue(about: str, issue: str, shown: list = ()) -> dict:
    return {"about": about, "issue": issue, "shown_in": list(shown)}


def _how_sure(judged: dict, category: str) -> str:
    return f"{round(100 * (judged.get('probabilities') or {}).get(category, 0))}% sure"


def _how_sure_of_the_title(title: dict) -> str:
    """The title's own probability; a "same job" also takes the share Jev gave "exact", as title_judged counts it."""
    probabilities = title.get("probabilities") or {}
    sure = probabilities.get(title["category"], 0) + (probabilities.get("exact", 0) if title["category"] == "same_job" else 0)
    return f"{round(100 * sure)}% sure"


def years_of_all_roles(resume_text: str, today=None) -> float:
    """The years the résumé's roles cover together, an overlap counted once."""
    months = set()
    for role in ats_text.roles_in(resume_text, today):
        months |= role["months"]
    return round(len(months) / 12, 1)
