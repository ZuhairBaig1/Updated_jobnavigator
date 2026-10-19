"""ATS score, Tailor + Jev: a focused copy's own requirement list, judged by Jev, with the issues it finds in plain
words. No test here calls a model."""
from datetime import date

import pytest

from backend.analyzer import ats_score_fast_jev as fast
from backend.analyzer import ats_score_tailor_jev as tailor_jev
from backend.analyzer import ats_text
from backend.analyzer.cv_scorer import resume_text_for_scoring
from backend.models.db import Resume, Setting

TODAY = date(2026, 10, 19)
RESUME = resume_text_for_scoring({
    "header": {"name": "Priya Raman"},
    "summary": "Backend developer.",
    "experience": [{"company": "Acme", "title": "Java Developer", "date": "Jan 2024 – Present",
                    "bullets": ["Built Spring Boot services on Kafka, cutting latency 30%"]}],
    "skills": {"Backend": "Java, Kafka, Docker"},
})
POSTING = ("Senior Java Developer\nRequirements\n- Strong Kafka in production\n- Docker and Kubernetes\n"
           "- 5+ years of Java experience\n- Excellent communication\n- Terraform is a plus\n"
           "Duties\n- Track work in Jira\n- Mentor junior engineers")
KAFKA_BULLET = "Built Spring Boot services on Kafka, cutting latency 30%"
REPORT = {
    "requirements": [{"name": "Kafka", "kind": "must"}, {"name": "Docker", "kind": "must"},
                     {"name": "Kubernetes", "kind": "must"}, {"name": "5+ years of Java experience", "kind": "must"},
                     {"name": "communication", "kind": "must"}, {"name": "Event streaming", "kind": "must"},
                     {"name": "Terraform", "kind": "nice"}, {"name": "Jira", "kind": "duty"},
                     {"name": "Mentor junior engineers", "kind": "duty"}],
    "requirements_met": [{"requirement": "Kafka", "met": True, "found_at": {},
                          "shown_in": [{"where": "Java Developer", "text": KAFKA_BULLET},
                                       {"where": "Java Developer", "text": "A bullet edited away after tailoring"}]},
                         {"requirement": "Event streaming", "met": True, "found_at": {},
                          "shown_in": [{"where": "Java Developer", "text": KAFKA_BULLET}]},
                         {"requirement": "Mentor junior engineers", "met": True, "found_at": {},
                          "shown_in": [{"where": "Java Developer", "text": KAFKA_BULLET}]}],
    "other_names": {},
}
# Jev's evidence level for each requirement: 0 none, 1 listed, 2 partly, 3 shown, 4 shown with a result.
EVIDENCE = {"Kafka": 4, "Docker": 1, "Kubernetes": 0, "Event streaming": 2, "Terraform": 0, "Jira": 0,
            "Mentor junior engineers": 0, "communication": 0}


def _fake_jev(traits=("communication",)):
    async def fake_post(client, route, state, questions):
        answers = {}
        for key, question in questions.items():
            named = question["instructions"].split('"')[1] if '"' in question["instructions"] else ""
            if question["type"] == "score":
                answers[key] = {"probabilities": {str(EVIDENCE[named]): 1.0}, "confidence": 0.9}
            elif key.startswith("trait"):
                answers[key] = {"noul": 0.9 if named in traits else 0.1}
            elif question["type"] == "noul":
                answers[key] = {"noul": 0.9 if key == "seniority stated" else 0.1}
            else:
                pick = {"occupation": "same", "seniority": "far_below", "title": "related"}[key.rsplit(" ", 1)[0]]
                answers[key] = {"choice": pick, "probabilities": {pick: 1.0}, "confidence": 0.9}
        return {"answers": answers, "usage": {"input_tokens": 1000}, "model": "jev-1.13.0"}
    return fake_post


@pytest.fixture
def jev(monkeypatch):
    async def no_extractor(*args, **options):
        raise AssertionError("Tailor + Jev asks no model but Jev")

    monkeypatch.setattr(fast, "call_llm", no_extractor)
    monkeypatch.setattr(fast, "_post_to_jev", _fake_jev())
    monkeypatch.setattr(fast, "jev_route", lambda: {"url": "u", "model": "m", "key": "k", "provider": "openrouter"})
    tailor_jev._scores_remembered.clear()


# ── The requirement list, from the tailoring ───────────────────────────────────

def test_the_tailorings_list_gets_its_posting_lines_and_terms_and_years_are_left_to_seniority():
    _, posting_lines = ats_text.numbered_posting(POSTING)
    listed, years = tailor_jev.requirements_from_the_tailoring(REPORT, posting_lines)
    by_name = {item["requirement"]: item for item in listed}
    assert years == ["5+ years of Java experience"] and "5+ years of Java experience" not in by_name
    assert by_name["Kafka"]["terms"] == ["Kafka"] and "Kafka" in by_name["Kafka"]["line_text"]
    assert by_name["Event streaming"]["terms"] == [] and by_name["Event streaming"]["line"] == ""   # not in the posting
    assert by_name["Jira"]["terms"] == ["Jira"]                          # a duty that names a tool keeps its term
    assert by_name["Mentor junior engineers"]["terms"] == []             # a duty that names none has none


def test_a_pair_the_tailoring_found_counts_as_another_name():
    item = {"requirement": "MSSQL", "wording": "none", "terms": ["MSSQL"]}
    judged = tailor_jev.with_the_tailorings_other_names([item], {"MSSQL": "SQL Server"}, "Built reports on SQL Server.")
    assert judged[0]["wording"] == "other"


# ── The whole score ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_jev_alone_scores_the_tailorings_list_and_traits_are_left_out(jev):
    result = await tailor_jev.ats_score(RESUME, POSTING, REPORT, posting_title="Senior Java Developer", today=TODAY)
    assert result["scorer"] == "ats_tailor_jev" and 0 <= result["ats_score"] <= 100
    assert result["left_out"] == {"years": ["5+ years of Java experience"], "traits": ["communication"]}
    assert "communication" not in [item["requirement"] for item in result["requirements"]]
    assert set(result["timings"]) == {"judge_overall", "judge_requirements", "total"}
    assert result["cost_usd"] == round(2000 * fast.JEV_DOLLARS_PER_MILLION_INPUT_TOKENS / 1e6, 6)
    again = await tailor_jev.ats_score(RESUME, POSTING, REPORT, posting_title="Senior Java Developer", today=TODAY)
    assert again["remembered"] is True and again["ats_score"] == result["ats_score"]


@pytest.mark.asyncio
async def test_each_requirement_is_shown_in_the_tailorings_lines_then_the_lines_writing_its_term(jev):
    result = await tailor_jev.ats_score(RESUME, POSTING, REPORT, posting_title="Senior Java Developer", today=TODAY)
    kafka = next(item for item in result["requirements"] if item["requirement"] == "Kafka")
    # The tailoring's line first; a line the copy no longer has is left out; the same line found twice is shown once.
    assert kafka["cited_text"] == [f"Java Developer: {KAFKA_BULLET}", "Skills: Backend: Java, Kafka, Docker"]
    assert [line["found_by"] for line in kafka["shown_in"]] == ["tailoring", "search term"]


@pytest.mark.asyncio
async def test_the_issues_are_written_from_the_answers_in_the_score(jev):
    result = await tailor_jev.ats_score(RESUME, POSTING, REPORT, posting_title="Senior Java Developer", today=TODAY)
    issues = {issue["issue"]: issue for issue in result["issues"]}
    assert "Kubernetes: not in your résumé." in issues
    listed = issues["Docker: only listed (skills, summary or education); no role or project shows you using it."]
    assert listed["shown_in"] == [{"where": "Skills", "text": "Backend: Java, Kafka, Docker", "found_by": "search term"}]
    partly = issues["Event streaming: a role or project shows only part of it, or something close to it."]
    assert partly["shown_in"][0]["text"] == KAFKA_BULLET
    assert not any(issue.startswith(("Kafka", "communication")) for issue in issues)     # shown, or not scored
    assert "Not in your résumé: Terraform." in issues
    experience = next(issue for issue in issues if issue.startswith("The posting asks for"))
    assert "5+ years of Java experience" in experience and "about 2.8 years" in experience and "well short" in experience
    assert any(issue.startswith("Your résumé does not write the posting's title \"Senior Java Developer\"") for issue in issues)
    assert "Not scored: personal traits (communication); years of experience (seniority covers them)." in issues
    assert not any(issue["about"] in ("occupation", "format") for issue in result["issues"])   # same occupation; format has its own table
    assert [issue["about"] for issue in result["issues"]][:3] == ["must-have"] * 3                 # the must-haves first


@pytest.mark.asyncio
async def test_a_must_have_jev_finds_nowhere_but_the_tailoring_points_to_is_not_called_missing(jev, monkeypatch):
    monkeypatch.setitem(EVIDENCE, "Kafka", 0)
    result = await tailor_jev.ats_score(RESUME, POSTING, REPORT, posting_title="Senior Java Developer", today=TODAY)
    issue = next(issue for issue in result["issues"] if issue["issue"].startswith("Kafka"))
    # Kafka is written in the role, so code is sure it is there; Jev's "none" is not repeated as "not in your résumé".
    assert issue["issue"] == "Kafka: named in a role, but no bullet shows you doing it."
    monkeypatch.setitem(EVIDENCE, "Event streaming", 0)
    tailor_jev._scores_remembered.clear()
    result = await tailor_jev.ats_score(RESUME, POSTING, REPORT, posting_title="Senior Java Developer", today=TODAY)
    issue = next(issue for issue in result["issues"] if issue["issue"].startswith("Event streaming"))
    assert issue["issue"] == "Event streaming: Jev found nothing, though the tailoring points to the line below; check it."
    assert issue["shown_in"][0]["text"] == KAFKA_BULLET


# ── The route ──────────────────────────────────────────────────────────────────

def _a_copy(test_db, json_data):
    test_db.add(Setting(key="dashboard_api_key", value=""))
    resume = Resume(name="Copy", is_base=False, template="inter", json_data={
        "header": {"name": "Priya Raman", "title": "Senior Java Developer", "target_title": "Senior Java Developer"},
        "summary": "Backend engineer with Java and Kafka work for years.",
        "experience": [{"company": "Acme", "title": "Java Developer", "date": "Jan 2024 – Present",
                        "bullets": [KAFKA_BULLET]}],
        "_tailor_context": {"job_description": POSTING}, **json_data})
    test_db.add(resume)
    test_db.commit()
    return resume


def test_the_route_scores_a_focused_copy_with_its_report_and_target_title(api_client, test_db, monkeypatch):
    resume = _a_copy(test_db, {"_focused_report": REPORT})
    seen = {}

    async def fake_ats_score(resume_text, job_text, report, title_lines=(), posting_title=""):
        seen.update(job_text=job_text, report=report, posting_title=posting_title)
        return {"ats_score": 64, "scorer": "ats_tailor_jev"}

    monkeypatch.setattr(tailor_jev, "ats_score", fake_ats_score)
    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score", json={"scorer": "ats_tailor_jev"})
    assert reply.status_code == 200, reply.text
    assert seen == {"job_text": POSTING, "report": REPORT, "posting_title": "Senior Java Developer"}


TAILORED_REPLY = {
    "target_job_title": "Senior Java Developer", "header": {"location": "", "work_authorization": ""},
    "requirements": [{"name": "Kafka", "kind": "must", "has": True, "by": [{"source": "job", "at": "1.1"}]}],
    "same_thing_other_name": [], "professional_summary": "Java developer who builds Spring Boot services on Kafka.",
    "technical_skills": [{"category": "Backend", "skills": "Java, Kafka"}],
    "work_experience": [{"role_number": 1, "bullets": [{"text": KAFKA_BULLET, "from_bullets": [1]}]}],
    "projects": [], "certifications": [],
    "requirements_met": [{"name": "Kafka", "met": True, "by": [{"source": "job", "at": "1.1"}]}],
}


def _a_base_and_the_tailor(test_db, monkeypatch, ats_score):
    import json
    test_db.add(Setting(key="tailor_auto_quick_score", value="off"))
    test_db.add(Setting(key="cv_tailor_prompt", value="p {resume_json} {job_description}"))   # read in every mode
    base = Resume(name="Base", is_base=True, template="inter", json_data={
        "header": {"name": "Priya Raman", "title": "Java Developer"}, "summary": "Java developer.",
        "skills": {"Backend": "Java, Kafka"},
        "experience": [{"title": "Java Developer", "company": "Acme", "start": "2024-01", "end": "", "current": True,
                        "bullets": [KAFKA_BULLET]}]})
    test_db.add(base)
    test_db.commit()

    async def fake_llm(prompt, system, max_tokens=3000, **kwargs):
        return {"text": json.dumps(TAILORED_REPLY), "usage": {}, "provider": "openrouter", "model": "openai/gpt-6-luna"}

    monkeypatch.setattr("backend.analyzer.llm_client.call_cv_tailor_llm", fake_llm)
    monkeypatch.setattr(tailor_jev, "ats_score", ats_score)
    return str(base.id)


@pytest.mark.asyncio
async def test_a_focused_copy_is_scored_by_tailor_jev_the_moment_it_is_made(test_db, monkeypatch):
    import backend.api.routes_resumes as rr
    scored = []

    async def fake_ats_score(resume_text, job_text, report, title_lines=(), posting_title=""):
        scored.append({"job_text": job_text, "report": report, "posting_title": posting_title})
        return {"ats_score": 61, "scorer": "ats_tailor_jev"}

    base_id = _a_base_and_the_tailor(test_db, monkeypatch, fake_ats_score)
    summary = await rr._tailor_impl(base_id, None, POSTING, mode="focused")
    assert "Created" in summary and "Tailor + Jev score 61" in summary
    assert scored[0]["job_text"] == POSTING and scored[0]["report"]["requirements"] == [{"name": "Kafka", "kind": "must"}]
    assert scored[0]["posting_title"] == "Senior Java Developer"

    scored.clear()
    summary = await rr._tailor_impl(base_id, None, POSTING, mode="complete")
    assert "Created" in summary and scored == []                     # a complete copy keeps no list to score


@pytest.mark.asyncio
async def test_a_failed_score_costs_the_tailored_copy_nothing(test_db, monkeypatch):
    import backend.api.routes_resumes as rr

    async def failing_ats_score(*args, **kwargs):
        raise RuntimeError("Jev is down")

    base_id = _a_base_and_the_tailor(test_db, monkeypatch, failing_ats_score)
    summary = await rr._tailor_impl(base_id, None, POSTING, mode="focused")
    assert "Created" in summary and "Tailor + Jev score" not in summary
    assert test_db.query(Resume).filter(Resume.is_base == False).count() == 1   # noqa: E712


@pytest.mark.parametrize("json_data, job_description", [
    ({}, ""),                                                    # not a focused copy
    ({"_focused_report": {"not_supported": []}}, ""),            # a focused copy made before the list was kept
    ({"_focused_report": REPORT}, "Python developer wanted"),    # another posting than its own
])
def test_the_route_refuses_what_tailor_jev_cannot_score(api_client, test_db, json_data, job_description):
    resume = _a_copy(test_db, json_data)
    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score",
                            json={"scorer": "ats_tailor_jev", "job_description": job_description})
    assert reply.status_code == 400 and "Tailor + Jev scores a focused copy" in reply.json()["detail"]
