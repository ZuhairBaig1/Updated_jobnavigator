"""The exact JSON the parser or the tailor replied with is kept beside the résumé it made (Resume.model_reply), so the
app can show it as the model gave it. No test here calls a model."""
import json

import pytest

from backend.models.db import Resume, Setting

PARSER_REPLY = {
    "header": {"name": "Jordan Lee", "title": "Data Engineer", "location": "", "email": "", "phone": "",
               "work_authorization": "", "github": "", "linkedin": ""},
    "professional_summary": "Data engineer building batch pipelines.", "technical_skills": [], "projects": [],
    "education": [], "certifications": [],
    "work_experience": [{"job_title": "Data Engineer", "company": "Acme", "location": "", "dates": "",
                         "start": "2020-01", "end": "", "current": True, "bullets": ["Built Airflow pipelines."]}],
}
POSTING = "Senior Java Developer\nRequirements\n- Strong Kafka in production"
BULLET = "Built Spring Boot services on Kafka, cutting latency 30%"
FOCUSED_REPLY = {
    "target_job_title": "Senior Java Developer", "header": {"location": "", "work_authorization": ""},
    "requirements": [{"name": "Kafka", "kind": "must", "has": True, "by": [{"source": "job", "at": "1.1"}]}],
    "same_thing_other_name": [], "professional_summary": "Java developer who builds Spring Boot services on Kafka.",
    "technical_skills": [{"category": "Backend", "skills": "Java, Kafka"}],
    "work_experience": [{"role_number": 1, "bullets": [{"text": BULLET, "from_bullets": [1]}]}],
    "projects": [], "certifications": [],
    "requirements_met": [{"name": "Kafka", "met": True, "by": [{"source": "job", "at": "1.1"}]}],
}


def test_an_imported_resume_keeps_the_parsers_exact_reply(api_client, test_db, monkeypatch):
    import backend.analyzer.llm_client as llm_client
    import backend.api.routes_resumes as rr
    test_db.add(Setting(key="dashboard_api_key", value=""))
    test_db.commit()

    async def extracted(pdf_bytes):
        return "Jordan Lee\nData Engineer at Acme, 2020 – Present\n- Built Airflow pipelines.\n" * 3

    async def fake_parser(**options):
        return {"text": json.dumps(PARSER_REPLY), "usage": {"input_tokens": 1200, "output_tokens": 800},
                "provider": "openrouter", "model": "google/gemini-3-flash"}

    monkeypatch.setattr(rr, "_extract_pdf_text", extracted)
    monkeypatch.setattr(llm_client, "call_llm", fake_parser)
    monkeypatch.setattr(llm_client, "resolve_llm_config",
                        lambda purpose, db=None: {"provider": "openrouter", "model": "google/gemini-3-flash", "api_key": "k"})
    made = api_client.post("/api/resumes/import-pdf", files={"file": ("Jordan.pdf", b"%PDF-1.4 a resume", "application/pdf")})
    assert made.status_code == 201, made.text

    shown = api_client.get(f"/api/resumes/{made.json()['id']}").json()
    kept = shown["model_reply"]
    assert kept["reply"] == PARSER_REPLY                                   # exactly as the model wrote it
    assert (kept["step"], kept["model"], kept["input_tokens"], kept["output_tokens"]) == \
        ("parser", "google/gemini-3-flash", 1200, 800)
    assert "model_reply" not in shown["json_data"]                         # never mixed into the résumé itself
    assert shown["json_data"]["experience"][0]["bullets"] == ["Built Airflow pipelines."]   # still processed as before


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, reply", [("focused", FOCUSED_REPLY), ("", {"summary": "Tailored for the Kafka work."})])
async def test_a_tailored_copy_keeps_the_tailors_exact_reply(test_db, monkeypatch, mode, reply):
    import backend.api.routes_resumes as rr
    test_db.add(Setting(key="tailor_auto_quick_score", value="off"))
    test_db.add(Setting(key="cv_tailor_prompt", value="p {resume_json} {job_description}"))
    base = Resume(name="Base", is_base=True, template="inter", json_data={
        "header": {"name": "Priya Raman", "title": "Java Developer"}, "summary": "Java developer.",
        "skills": {"Backend": "Java, Kafka"},
        "experience": [{"title": "Java Developer", "company": "Acme", "start": "2024-01", "end": "", "current": True,
                        "bullets": [BULLET]}]})
    test_db.add(base)
    test_db.commit()

    async def fake_tailor(prompt, system, max_tokens=3000, **options):
        return {"text": json.dumps(reply), "usage": {"input_tokens": 5559, "output_tokens": 5304},
                "provider": "openrouter", "model": "openai/gpt-6-luna"}

    async def no_score(resume_id, asked_for=""):
        return None

    monkeypatch.setattr("backend.analyzer.llm_client.call_cv_tailor_llm", fake_tailor)
    monkeypatch.setattr(rr, "_tailor_jev_score", no_score)
    await rr._tailor_impl(str(base.id), None, POSTING, mode=mode)

    test_db.expire_all()
    copy = test_db.query(Resume).filter(Resume.is_base == False).one()       # noqa: E712
    assert copy.model_reply["reply"] == reply                                # before the app's checks changed anything
    assert (copy.model_reply["step"], copy.model_reply["mode"], copy.model_reply["model"]) == \
        ("tailor", mode or "full", "openai/gpt-6-luna")
    assert copy.model_reply["output_tokens"] == 5304 and "model_reply" not in copy.json_data
