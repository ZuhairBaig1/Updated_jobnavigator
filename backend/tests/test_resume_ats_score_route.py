"""The ATS score of a résumé, served for the frontends."""
from backend.analyzer import line_match
from backend.models.db import Resume, Setting

SCORE = {"ats_score": 72, "coverage_score": 76, "same_occupation": 0.95, "must_have_coverage": 0.8,
         "preferred_coverage": 0.6, "found_from": "headings",
         "must_have": [{"line": "Kubernetes", "covered": 1.0, "weight": 1.0, "importance": 3.5}],
         "preferred": [{"line": "Terraform", "covered": 0.6, "weight": 1.0, "importance": 2.0}]}


def _resume(**extra):
    return Resume(name="DevOps", is_base=not extra, template="inter", json_data={
        "summary": "Senior DevOps engineer with ten years of platform work on Kubernetes",
        "experience": [{"company": "Acme", "title": "DevOps Engineer", "bullets": ["Ran Kubernetes clusters"]}],
        "skills": {"core": "Kubernetes, Terraform"}, **extra})


def _saved(test_db, resume=None):
    """Save the résumé, with the dashboard open to requests that carry no key."""
    test_db.add(Setting(key="dashboard_api_key", value=""))
    if resume is not None:
        test_db.add(resume)
    test_db.commit()
    return resume


def _scores_are(monkeypatch, result):
    asked = []

    async def fake_ats_score(scorer, resume_text, job_text, title_lines=()):
        asked.append({"scorer": scorer, "resume_text": resume_text, "job_text": job_text,
                      "title_lines": title_lines})
        return result
    monkeypatch.setattr(line_match, "ats_score", fake_ats_score)
    return asked


def test_a_resume_is_scored_against_the_pasted_job_description(api_client, test_db, monkeypatch):
    resume = _saved(test_db, _resume())
    asked = _scores_are(monkeypatch, SCORE)

    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score", json={"job_description": "Requirements\n- Kubernetes"})

    assert reply.status_code == 200, reply.text
    assert reply.json() == SCORE
    assert asked[0]["job_text"] == "Requirements\n- Kubernetes"
    assert "Ran Kubernetes clusters" in asked[0]["resume_text"]


def test_a_tailored_copy_is_scored_against_the_description_it_was_tailored_for(api_client, test_db, monkeypatch):
    copy = _saved(test_db, _resume(_tailor_context={"job_description": "Senior DevOps Engineer posting",
                                                    "source": "freeform"}))
    asked = _scores_are(monkeypatch, SCORE)

    reply = api_client.post(f"/api/resumes/{copy.id}/ats-score", json={})

    assert reply.status_code == 200, reply.text
    assert asked[0]["job_text"] == "Senior DevOps Engineer posting"


def test_a_resume_with_no_job_description_to_score_against_is_refused(api_client, test_db, monkeypatch):
    resume = _saved(test_db, _resume())
    _scores_are(monkeypatch, SCORE)

    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score", json={})

    assert reply.status_code == 400
    assert "job description" in reply.json()["detail"]


def test_a_scoring_model_that_cannot_be_reached_is_reported(api_client, test_db, monkeypatch):
    resume = _saved(test_db, _resume())
    _scores_are(monkeypatch, None)

    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score", json={"job_description": "Requirements\n- Kubernetes"})

    assert reply.status_code == 503


def test_a_resume_that_does_not_exist_is_not_found(api_client, test_db, monkeypatch):
    _saved(test_db)
    _scores_are(monkeypatch, SCORE)

    reply = api_client.post("/api/resumes/00000000-0000-0000-0000-000000000000/ats-score", json={"job_description": "x"})

    assert reply.status_code == 404


def test_the_title_lines_under_the_name_are_sent_to_be_matched_against_the_posting(api_client, test_db, monkeypatch):
    saved = _saved(test_db, _resume(header={"name": "Jordan Lee", "target_title": "Senior DevOps Engineer",
                                            "title": "Site Reliability Engineer"}))
    asked = _scores_are(monkeypatch, SCORE)

    reply = api_client.post(f"/api/resumes/{saved.id}/ats-score", json={"job_description": "Requirements\n- Kubernetes"})

    assert reply.status_code == 200, reply.text
    assert asked[0]["title_lines"] == ["Senior DevOps Engineer", "Site Reliability Engineer"]


def test_the_rubric_scorer_is_used_when_asked_for_on_the_model_named(api_client, test_db, monkeypatch):
    from backend.analyzer import rubric_ats_score
    resume = _saved(test_db, _resume())
    asked = []

    async def fake_rubric_score(resume_text, job_text, provider, model):
        asked.append((provider, model, job_text))
        return {"ats_score": 81, "scorer": "rubric"}
    monkeypatch.setattr(rubric_ats_score, "ats_score", fake_rubric_score)

    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score",
                            json={"job_description": "Requirements\n- Kubernetes", "scorer": "rubric", "model": "haiku"})

    assert reply.status_code == 200, reply.text
    assert reply.json() == {"ats_score": 81, "scorer": "rubric"}
    assert asked == [("openrouter", "anthropic/claude-haiku-5.5", "Requirements\n- Kubernetes")]


def test_a_scoring_model_that_is_not_offered_is_refused(api_client, test_db):
    resume = _saved(test_db, _resume())

    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score",
                            json={"job_description": "Requirements\n- Kubernetes", "scorer": "rubric", "model": "gpt"})

    assert reply.status_code == 400


def test_an_account_out_of_credit_is_said_in_plain_words(api_client, test_db, monkeypatch):
    from backend.api import routes_resumes
    resume = _saved(test_db, _resume())
    _scores_are(monkeypatch, None)

    async def out_of_credit():
        return True
    monkeypatch.setattr(routes_resumes, "_openrouter_is_out_of_credit", out_of_credit)

    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score", json={"job_description": "Requirements\n- Kubernetes"})

    assert reply.status_code == 402
    assert "out of credit" in reply.json()["detail"]


def test_a_failure_on_the_claude_plan_is_never_blamed_on_openrouter_credit(api_client, test_db, monkeypatch):
    from backend.analyzer import rubric_ats_score
    from backend.api import routes_resumes
    resume = _saved(test_db, _resume())

    async def no_score(resume_text, job_text, provider, model):
        return None

    async def out_of_credit():
        return True
    monkeypatch.setattr(rubric_ats_score, "ats_score", no_score)
    monkeypatch.setattr(routes_resumes, "_openrouter_is_out_of_credit", out_of_credit)

    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score",
                            json={"job_description": "Requirements\n- Kubernetes", "scorer": "rubric", "model": "haiku_plan"})

    assert reply.status_code == 503
