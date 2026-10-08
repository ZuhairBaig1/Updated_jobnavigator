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

    async def fake_ats_score(scorer, resume_text, job_text):
        asked.append({"scorer": scorer, "resume_text": resume_text, "job_text": job_text})
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

    assert reply.status_code == 502


def test_a_resume_that_does_not_exist_is_not_found(api_client, test_db, monkeypatch):
    _saved(test_db)
    _scores_are(monkeypatch, SCORE)

    reply = api_client.post("/api/resumes/00000000-0000-0000-0000-000000000000/ats-score", json={"job_description": "x"})

    assert reply.status_code == 404
