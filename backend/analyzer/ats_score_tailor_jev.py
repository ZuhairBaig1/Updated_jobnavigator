"""ATS score, Tailor + Jev: a focused copy's own requirement list, judged by Jev.

(Built 2026-10-19 from the Tailor + Jev experiment; see the ATS score rules.)

Fast (Jev) first asks a small model for the posting's requirements, their search terms and the posting's title, then
Jev judges them. A focused copy already carries all three from its tailoring, so that call is skipped and Jev is the
only model asked:
  - The requirements are the tailoring's list, kept in the copy's report (focused_tailor.py). Each is given the
    posting line that writes its name, and its name is its search term when that line writes it, as Fast keeps only
    the terms their line writes. A duty has a term only when its name is a tool the skills dictionary knows (Jira),
    as Fast gives none to a duty that names no tool.
  - Years of experience are left out, as Fast's extractor is told to (seniority covers them), and so are personal
    traits. Whether an entry is a trait is a question of meaning, so Jev is asked it beside the other questions.
  - A pair the tailoring found to be the same thing under two names counts as another name, as does Jev's own
    "another name?" answer.
  - The posting's title is the copy's target title.
Everything after that is Fast (Jev)'s own code: the same Jev questions, keyword matching and arithmetic, so it
scores on Default's scale. Then ats_score_issues.py says in plain words what the score found, and where the copy
shows each requirement, which Jev cannot write.

It scores a focused copy against the posting it was tailored for. A copy tailored before 2026-10-19 has no
requirement list in its report and cannot be scored this way.

Measured 2026-10-19 on 16 copies of 4 résumés: about 1.1 s and $0.0008 a score (Fast: 6.4 s, $0.0016); its own
posting scored 30 points above a mismatched one on average (Fast 33.5, Default 29), and a score moved about
1 point between runs (Fast 2, Default 6).
"""
import asyncio
import re
import time
from datetime import date

from backend.analyzer import ats_score_default, ats_score_issues, ats_text, posting_keywords, resume_format_checks
from backend.analyzer import ats_score_fast_jev as fast

VERSION = "tailor-jev-1.0"
TRAIT_THRESHOLD = 0.5
MOST_SCORES_REMEMBERED = 500
# A requirement that is a length of time ("4-6 years of overall industry experience"): seniority covers it.
A_LENGTH_OF_TIME = re.compile(r"\d.*\byears?\b", re.IGNORECASE)

_scores_remembered = {}


async def ats_score(resume_text: str, job_text: str, report: dict, title_lines=(), posting_title: str = "",
                    today: date | None = None) -> dict | None:
    """The score out of 100 with every requirement, term, judgement and issue behind it, or None when Jev cannot be
    reached. report is the focused copy's report, holding the tailoring's requirements and requirements_met."""
    started = time.monotonic()
    remembered_as = fast._key_of(VERSION, resume_text, job_text, report, title_lines, posting_title, today)
    if remembered_as in _scores_remembered:
        return {**_scores_remembered[remembered_as], "remembered": True}

    resume_for_the_model, removed_from_the_resume = ats_text.without_lines_to_the_evaluator(resume_text)
    job_for_the_model, removed_from_the_posting = ats_text.without_lines_to_the_evaluator(job_text)
    model_text = ats_text.resume_for_the_model(resume_for_the_model, title_lines, today)
    posting, posting_lines = ats_text.numbered_posting(job_for_the_model)
    state = fast.jev_state(posting, model_text)

    listed, years_asked = requirements_from_the_tailoring(report, posting_lines)
    by_requirement, overall = await asyncio.gather(
        fast.ask_jev(state, {**fast.requirement_questions(listed), **trait_questions(listed)}),
        fast.ask_jev(state, fast.overall_questions()))
    if by_requirement is None or overall is None:
        return None

    judged = fast.requirements_judged(listed, by_requirement["answers"], resume_for_the_model)
    judged = with_the_tailorings_other_names(judged, report.get("other_names") or {}, resume_for_the_model)
    requirements, traits = [], []
    for number, item in enumerate(judged, start=1):
        trait = float((by_requirement["answers"].get(f"trait {number}") or {}).get("noul", 0) or 0)
        (traits if trait >= TRAIT_THRESHOLD else requirements).append(item)
    keywords = posting_keywords.keyword_part(requirements, resume_for_the_model)
    occupation = fast.choice_judged(overall["answers"], "occupation", ats_score_default.OCCUPATION_SHARE)
    seniority = fast.seniority_judged(overall["answers"])
    title = fast.title_judged(overall["answers"], posting_title,
                              ats_score_default.resume_title_text(resume_for_the_model, title_lines))
    score = ats_score_default.score_from_parts(requirements, keywords["share"], seniority["share"], title["share"],
                                               occupation["share"])

    for item in requirements:
        item["shown_in"] = ats_score_issues.shown_in(item, report.get("requirements_met"), resume_for_the_model)
        item["cited_text"] = [f"{line['where']}: {line['text']}" for line in item["shown_in"]]
    left_out = {"years": years_asked, "traits": [item["requirement"] for item in traits]}
    issues = ats_score_issues.issues_of(requirements, keywords, seniority, title, occupation, left_out,
                                        resume_for_the_model, today)
    input_tokens = by_requirement["input_tokens"] + overall["input_tokens"]
    result = {**score, "scorer": "ats_tailor_jev", "version": VERSION, "model": by_requirement["model"],
              "requirements": requirements, "keywords": keywords,
              "occupation": occupation, "seniority": seniority, "title": title,
              "issues": issues, "left_out": left_out,
              "missing": ats_score_default.missing_from(requirements), "least_certain": fast.least_certain(requirements),
              "format": resume_format_checks.checks(resume_text),
              "warnings": ats_text.length_warnings(resume_text, job_text),
              "lines_removed": {"resume": removed_from_the_resume, "posting": removed_from_the_posting},
              "remembered": False,
              "timings": {"judge_overall": overall["seconds"], "judge_requirements": by_requirement["seconds"],
                          "total": round(time.monotonic() - started, 2)},
              "seconds": round(time.monotonic() - started, 2),
              "input_tokens": input_tokens, "output_tokens": 0,
              "cost_usd": round(input_tokens * fast.JEV_DOLLARS_PER_MILLION_INPUT_TOKENS / 1e6, 6)}
    fast._remember(_scores_remembered, remembered_as, result, MOST_SCORES_REMEMBERED)
    return result


def requirements_from_the_tailoring(report: dict, posting_lines: dict) -> tuple:
    """(the tailoring's requirements in the shape Fast's extractor gives, the names left out as years of experience)."""
    listed, years = [], []
    for entry in report.get("requirements") or []:
        name, kind = str(entry.get("name") or "").strip(), entry.get("kind")
        if not name or kind not in ats_score_default.KIND_WEIGHT:
            continue
        if A_LENGTH_OF_TIME.search(name):
            years.append(name)
            continue
        line = next((line_id for line_id, text in posting_lines.items() if posting_keywords.appears_in(name, text)), "")
        names_a_tool = kind != "duty" or posting_keywords.standard_name(name) is not None
        listed.append({"requirement": name, "line": line, "line_text": posting_lines.get(line, ""), "kind": kind,
                       "terms": [name] if line and names_a_tool else [], "note": ""})
    return listed[:ats_score_default.MOST_REQUIREMENTS], years


def trait_questions(requirements: list) -> dict:
    """Whether each entry is a personal trait, which Fast's extractor is told to leave out."""
    return {f"trait {number}": {"type": "noul", "instructions":
                                f"Is \"{item['requirement']}\" a personal trait, such as communication, ownership "
                                f"or attitude, rather than a skill, tool, qualification, field or duty?"}
            for number, item in enumerate(requirements, start=1)}


def with_the_tailorings_other_names(requirements: list, other_names: dict, resume_text: str) -> list:
    """A requirement the résumé does not word is worded "other" when the tailoring paired it with a name the résumé
    writes ({posting name: résumé name}, checked when the copy was made)."""
    resume_name_of = {posting_keywords.same_term_key(posting_name): resume_name
                      for posting_name, resume_name in other_names.items()}
    for item in requirements:
        resume_name = resume_name_of.get(posting_keywords.same_term_key(item["requirement"]))
        if item.get("wording") == "none" and resume_name and posting_keywords.appears_in(resume_name, resume_text):
            item["wording"] = "other"
    return requirements
