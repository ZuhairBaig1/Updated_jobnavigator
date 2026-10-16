"""ATS score, Fast (Jev): a fast model lists the requirements, Jev judges them. No test here calls a model."""
import asyncio
import json
import re
import time
from datetime import date

import pytest

from backend.analyzer import ats_score_fast_jev as fast
from backend.analyzer.cv_scorer import resume_text_for_scoring
from backend.models.db import Resume, Setting

TODAY = date(2026, 10, 15)
RESUME = resume_text_for_scoring({
    "header": {"name": "Priya Raman", "email": "priya@example.com", "phone": "+1 415 555 0134"},
    "experience": [{"company": "Acme", "title": "Senior Java Developer", "date": "Jan 2020 – Present",
                    "bullets": ["Built Spring Boot services on Kafka, cutting latency 30%"]}],
    "skills": {"Backend": "Java, Kafka"},
})
POSTING = "Senior Java Developer\nRequirements\n- Strong Kafka in production\n- Terraform is a plus"


# ── Step 1: the requirement list ───────────────────────────────────────────────

def test_the_extractor_reply_is_read_with_its_terms_checked_against_their_lines():
    reply = json.dumps({"posting_title": "Senior Java Developer", "requirements": [
        {"j": "[J3] Strong Kafka", "r": "Kafka", "k": "must", "s": ["Kafka", "event streaming"]},
        {"j": "J4", "r": "Rust", "k": "sometimes", "s": ["Rust"]}]})
    listed = fast.extracted_requirements(reply, {"J3": "Strong Kafka in production", "J4": "Terraform is a plus"})
    assert listed["posting_title"] == "Senior Java Developer"
    assert [item["requirement"] for item in listed["requirements"]] == ["Kafka"]
    assert listed["requirements"][0]["terms"] == ["Kafka"] and "event streaming" in listed["requirements"][0]["note"]


# ── Step 2: Jev's questions ────────────────────────────────────────────────────

def test_each_overall_choice_is_asked_in_both_option_orders():
    questions = fast.overall_questions()
    for name in ("occupation", "seniority", "title"):
        forward, backward = list(questions[f"{name} a"]["criteria"]), list(questions[f"{name} b"]["criteria"])
        assert backward == list(reversed(forward))
    assert questions["seniority stated"]["type"] == "noul"
    assert "not_stated" not in questions["seniority a"]["criteria"]


def test_one_evidence_score_per_requirement_and_another_name_only_for_one_with_terms():
    questions = fast.requirement_questions([
        {"requirement": "Kafka", "line_text": "Strong Kafka", "terms": ["Kafka"]},
        {"requirement": "Own services end to end", "line_text": "", "terms": []}])
    assert set(questions) == {"evidence 1", "other name 1", "evidence 2"}
    assert questions["evidence 1"]["type"] == "score" and len(questions["evidence 1"]["criteria"]) == 5
    assert "Kafka" in questions["evidence 1"]["instructions"]


@pytest.mark.asyncio
async def test_many_questions_go_in_parallel_requests_of_at_most_fifty(monkeypatch):
    sizes = []

    async def fake_post(client, route, state, questions):
        sizes.append(len(questions))
        return {"answers": {key: {"type": "noul", "noul": 0.9} for key in questions},
                "usage": {"input_tokens": 1000}, "model": "jev-1.13.0"}

    monkeypatch.setattr(fast, "_post_to_jev", fake_post)
    monkeypatch.setattr(fast, "jev_route", lambda: {"url": "u", "model": "m", "key": "k", "provider": "openrouter"})
    reply = await fast.ask_jev("state", {f"q{n}": {"type": "noul", "instructions": "?"} for n in range(120)})
    assert sorted(sizes) == [20, 50, 50] and len(reply["answers"]) == 120 and reply["input_tokens"] == 3000


# ── Step 3: Jev's answers into points ──────────────────────────────────────────

def test_points_are_the_expected_value_over_the_five_levels():
    answers = {"evidence 1": {"type": "score", "probabilities": {"0": 0, "1": 0, "2": 0, "3": 0.5, "4": 0.5},
                              "confidence": 0.4},
               "other name 1": {"noul": 0.9}}
    judged = fast.requirements_judged([{"requirement": "Terraform", "kind": "must", "terms": ["Terraform"],
                                      "line": "J4", "line_text": "", "note": ""}], answers, RESUME)[0]
    assert judged["points"] == pytest.approx(0.925) and judged["evidence"] == "shown"
    assert judged["wording"] == "other"          # not written, another name at 0.9 >= 0.7


def test_a_choice_asked_both_ways_averages_the_two():
    answers = {"occupation a": {"choice": "same", "probabilities": {"same": 0.8, "close": 0.2}, "confidence": 0.6},
               "occupation b": {"choice": "close", "probabilities": {"same": 0.4, "close": 0.6}, "confidence": 0.3}}
    judged = fast.choice_judged(answers, "occupation", {"same": 1.0, "close": 0.85, "neighbouring": 0.5,
                                                      "different": 0.15})
    assert judged["probabilities"]["same"] == 0.6 and judged["category"] == "same"
    assert judged["share"] == pytest.approx(0.6 * 1.0 + 0.4 * 0.85) and judged["orders_agree"] is False


def test_seniority_is_left_out_when_the_posting_states_no_years_or_level():
    answers = {"seniority stated": {"noul": 0.2},
               "seniority a": {"choice": "far_below", "probabilities": {"far_below": 1.0}}}
    assert fast.seniority_judged(answers)["share"] is None


def test_the_title_is_exact_only_when_written_word_for_word():
    answers = {"title a": {"choice": "exact", "probabilities": {"exact": 0.7, "same_job": 0.3}},
               "title b": {"choice": "exact", "probabilities": {"exact": 0.7, "same_job": 0.3}}}
    written = fast.title_judged(answers, "Senior Java Developer", "Senior Java Developer")
    not_written = fast.title_judged(answers, "Staff Engineer", "Senior Java Developer")
    assert written["share"] == 1.0 and written["category"] == "exact"
    assert not_written["category"] == "same_job" and not_written["share"] == pytest.approx(0.85)


def test_the_least_certain_requirement_is_reported():
    least = fast.least_certain([{"requirement": "Kafka", "confidence": 0.9}, {"requirement": "Go", "confidence": 0.3}])
    assert least["requirement"] == "Go" and least["below_threshold"] == 1


# ── Speed: the overall questions do not wait for the list; a posting's list is remembered ──

@pytest.mark.asyncio
async def test_the_overall_questions_run_while_the_list_is_made_and_the_list_is_remembered(monkeypatch):
    events, extractor_calls = [], []

    async def fake_call_llm(prompt, system, **options):
        extractor_calls.append(options)
        events.append(("extract started", time.monotonic()))
        await asyncio.sleep(0.2)
        events.append(("extract finished", time.monotonic()))
        kafka = re.search(r"\[(J\d+)\][^\n]*Kafka", prompt).group(1)
        return {"text": json.dumps({"posting_title": "Senior Java Developer", "requirements": [
            {"j": kafka, "r": "Kafka", "k": "must", "s": ["Kafka"]}]}), "usage": {"input_tokens": 500, "output_tokens": 60}}

    async def fake_post(client, route, state, questions):
        events.append(("jev " + ("overall" if "occupation a" in questions else "requirements"), time.monotonic()))
        answers = {}
        for key, question in questions.items():
            if question["type"] == "score":
                answers[key] = {"probabilities": {"4": 1.0}, "confidence": 1.0}
            elif question["type"] == "noul":
                answers[key] = {"noul": 0.9}
            else:
                best = next(iter(question["criteria"])) if key.endswith(" a") else list(question["criteria"])[-1]
                answers[key] = {"choice": best, "probabilities": {best: 1.0}, "confidence": 1.0}
        return {"answers": answers, "usage": {"input_tokens": 800}, "model": "jev-1.13.0"}

    monkeypatch.setattr(fast, "call_llm", fake_call_llm)
    monkeypatch.setattr(fast, "_post_to_jev", fake_post)
    monkeypatch.setattr(fast, "jev_route", lambda: {"url": "u", "model": "m", "key": "k", "provider": "openrouter"})
    fast._scores_remembered.clear()
    fast._postings_remembered.clear()

    first = await fast.ats_score(RESUME, POSTING, today=TODAY)
    when = dict(events)
    assert when["jev overall"] < when["extract finished"]             # did not wait for the list
    assert extractor_calls[0]["reasoning"] is False and extractor_calls[0]["model"] == fast.EXTRACTOR_MODEL
    assert first["scorer"] == "ats_fast_jev" and first["keywords"]["found"] == ["Kafka"]
    assert first["posting_remembered"] is False and set(first["timings"]) >= {"extract", "judge_overall", "total"}

    other_resume = RESUME.replace("cutting latency 30%", "cutting latency by a third")
    second = await fast.ats_score(other_resume, POSTING, today=TODAY)
    assert len(extractor_calls) == 1 and second["posting_remembered"] is True and second["timings"]["extract"] == 0.0


@pytest.mark.asyncio
async def test_no_key_for_jev_gives_no_score(monkeypatch):
    monkeypatch.setattr(fast, "jev_route", lambda: None)
    assert await fast.ask_jev("state", {"q": {"type": "noul", "instructions": "?"}}) is None


def test_the_route_scores_with_fast_jev_when_asked(api_client, test_db, monkeypatch):
    test_db.add(Setting(key="dashboard_api_key", value=""))
    resume = Resume(name="Java", is_base=True, template="inter", json_data={
        "header": {"name": "Priya Raman"}, "summary": "Backend engineer with Java and Kafka work for years.",
        "experience": [{"company": "Acme", "title": "Senior Java Developer", "date": "Jan 2020 – Present",
                        "bullets": ["Built Spring Boot services on Kafka"]}]})
    test_db.add(resume)
    test_db.commit()

    async def fake_ats_score(resume_text, job_text, title_lines=()):
        return {"ats_score": 71, "scorer": "ats_fast_jev"}

    monkeypatch.setattr(fast, "ats_score", fake_ats_score)
    reply = api_client.post(f"/api/resumes/{resume.id}/ats-score",
                            json={"job_description": "Java and Kafka", "scorer": "ats_fast_jev"})
    assert reply.status_code == 200, reply.text
    assert reply.json()["ats_score"] == 71
