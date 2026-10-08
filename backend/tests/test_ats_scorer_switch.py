"""The ats_scorer setting decides who produces the ATS score: Jev/GLiDE by default, the LLM as fallback."""
import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Imported before conftest swaps ats_scorer for its "always llm" stand-in.
from backend.analyzer.decision_scorer import ats_scorer as real_ats_scorer


class FakeJob:
    id = "job-1"
    company = "Acme"
    title = "Backend Engineer"
    description = "Job description here. " * 50
    cached_page_text = None
    url = None


def _decision_model_result(score):
    return {
        "score": score,
        "breakdown": {"keywords": 10, "domain": 20, "impact": 13, "experience": 15, "requirements": 3},
        "answers": {},
        "confidence": {},
        "usage": {"input_tokens": 100, "output_tokens": 5},
    }


def _llm_returning(text, calls):
    async def fake_call_llm(prompt, system, max_tokens, cached_prefix=None, **kwargs):
        calls.append(prompt)
        return {"text": text,
                "usage": {"input_tokens": 1, "output_tokens": 1,
                          "cache_read_tokens": 0, "cache_write_tokens": 0}}
    return fake_call_llm


@pytest.fixture
def scorer_db(monkeypatch):
    from backend.models.db import Setting
    engine = create_engine("sqlite:///:memory:")
    Setting.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    session.add(Setting(key="scoring_rubric", value="RUBRIC"))
    session.add(Setting(key="scoring_output_light", value='{"scores": {CV_NAMES_HERE}}'))
    session.add(Setting(key="scoring_output_full", value="FULL"))
    session.commit()
    session.close()

    from backend.analyzer import cv_scorer
    monkeypatch.setattr(cv_scorer, "SessionLocal", Session)
    monkeypatch.setattr(cv_scorer, "_get_scoring_semaphore", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr("backend.analyzer.cv_scorer.log_llm_call", lambda **kw: None)
    monkeypatch.setattr("backend.analyzer.decision_scorer.ats_scorer", lambda db=None: "jev")
    _use_requirements(monkeypatch, None)
    _use_question_set(monkeypatch, per_requirement=False)
    _use_posting_sections(monkeypatch, None)
    _use_judged_match(monkeypatch, {})
    return Session


def _use_judged_match(monkeypatch, result):
    """Stand in for the model's judgment of each keyword, which would otherwise call Jev."""
    async def fake_judged_match_for(scorer, resume_text, job_text, job):
        return result
    monkeypatch.setattr("backend.analyzer.cv_scorer._judged_match_for", fake_judged_match_for)


def _use_posting_sections(monkeypatch, sections):
    """Stand in for the posting's split into sections."""
    async def fake_sections_for_scoring(job, job_text):
        return sections
    monkeypatch.setattr("backend.analyzer.cv_scorer._sections_for_scoring", fake_sections_for_scoring)


def _use_question_set(monkeypatch, per_requirement):
    monkeypatch.setattr("backend.analyzer.decision_scorer.uses_per_requirement_questions",
                        lambda db=None: per_requirement)


def _use_requirements(monkeypatch, requirements):
    """Stand in for the requirements extraction, which would otherwise call a model."""
    async def fake_requirements_for_scoring(job, job_text):
        return requirements
    monkeypatch.setattr("backend.analyzer.cv_scorer._requirements_for_scoring", fake_requirements_for_scoring)


def _use_decision_model(monkeypatch, scores_by_resume_text):
    requirements_seen = []

    async def fake_score_resume(scorer, resume_text, job_text, requirements=None):
        requirements_seen.append(requirements)
        score = scores_by_resume_text.get(resume_text)
        return None if score is None else _decision_model_result(score)
    monkeypatch.setattr("backend.analyzer.decision_scorer.score_resume", fake_score_resume)
    return requirements_seen


async def _score(depth, cv_texts):
    from backend.analyzer import cv_scorer
    return await cv_scorer.score_job_sync(FakeJob(), cv_texts, db=None, depth=depth,
                                          preloaded_text="JD text")


@pytest.mark.asyncio
async def test_light_score_comes_from_jev_without_calling_the_llm(scorer_db, monkeypatch):
    llm_calls = []
    monkeypatch.setattr("backend.analyzer.cv_scorer.call_llm", _llm_returning("{}", llm_calls))
    _use_decision_model(monkeypatch, {"text A": 56, "text B": 70})

    result = await _score("light", {"A": "text A", "B": "text B"})

    assert result["scores"] == {"A": 56, "B": 70}
    assert result["best_cv"] == "B"
    assert llm_calls == []


@pytest.mark.asyncio
async def test_light_score_falls_back_to_the_llm_when_jev_fails(scorer_db, monkeypatch):
    llm_calls = []
    monkeypatch.setattr("backend.analyzer.cv_scorer.call_llm",
                        _llm_returning('{"scores": {"A": 78}, "best_cv": "A"}', llm_calls))
    _use_decision_model(monkeypatch, {})

    result = await _score("light", {"A": "text A"})

    assert result["scores"] == {"A": 78}
    assert len(llm_calls) == 1


@pytest.mark.asyncio
async def test_full_score_keeps_the_llm_report_but_uses_the_jev_score(scorer_db, monkeypatch):
    llm_reply = ('{"scores": {"A": 78}, "best_cv": "A", "summary": "Good fit.", '
                 '"breakdown": {"skills": 15, "experience": 18, "domain": 19, "role": 16, "requirements": 10}}')
    monkeypatch.setattr("backend.analyzer.cv_scorer.call_llm", _llm_returning(llm_reply, []))
    _use_decision_model(monkeypatch, {"text A": 56})

    result = await _score("full", {"A": "text A"})

    report = result["_scoring_report"]
    assert result["scores"] == {"A": 56}
    assert report["summary"] == "Good fit."
    assert report["breakdown"]["keywords"] == 10
    assert report["llm_scores"] == {"A": 78}
    assert report["llm_breakdown"]["skills"] == 15
    assert report["scored_by"] == "jev"


@pytest.mark.asyncio
async def test_llm_setting_skips_jev_entirely(scorer_db, monkeypatch):
    monkeypatch.setattr("backend.analyzer.decision_scorer.ats_scorer", lambda db=None: "llm")
    monkeypatch.setattr("backend.analyzer.cv_scorer.call_llm",
                        _llm_returning('{"scores": {"A": 78}, "best_cv": "A"}', []))

    async def jev_must_not_run(*args):
        raise AssertionError("Jev was called with the setting on llm")
    monkeypatch.setattr("backend.analyzer.decision_scorer.score_resume", jev_must_not_run)

    result = await _score("light", {"A": "text A"})

    assert result["scores"] == {"A": 78}


REQUIREMENTS = {"title": "Backend Engineer", "min_years": 5, "certifications": [], "responsibilities": [],
                "must_have": [{"name": "Python", "aliases": ["Py"], "quote": "Python"}],
                "nice_to_have": [{"name": "Kafka", "aliases": [], "quote": "Kafka"}]}


@pytest.mark.asyncio
async def test_every_resume_is_scored_against_the_extracted_requirements(scorer_db, monkeypatch):
    _use_requirements(monkeypatch, REQUIREMENTS)
    requirements_seen = _use_decision_model(monkeypatch, {"text A": 56, "text B": 70})

    await _score("light", {"A": "text A", "B": "text B"})

    assert requirements_seen == [REQUIREMENTS, REQUIREMENTS]


@pytest.mark.asyncio
async def test_the_per_requirement_question_set_scores_each_resume_by_requirement(scorer_db, monkeypatch):
    _use_requirements(monkeypatch, REQUIREMENTS)
    _use_question_set(monkeypatch, per_requirement=True)
    _use_decision_model(monkeypatch, {})

    async def fake_score_by_requirement(scorer, resume_text, requirements):
        return _decision_model_result(64)
    monkeypatch.setattr("backend.analyzer.decision_scorer.score_resume_by_requirement", fake_score_by_requirement)

    result = await _score("light", {"A": "text A"})

    assert result["scores"] == {"A": 64}


@pytest.mark.asyncio
async def test_the_five_questions_are_used_when_per_requirement_scoring_fails(scorer_db, monkeypatch):
    _use_requirements(monkeypatch, REQUIREMENTS)
    _use_question_set(monkeypatch, per_requirement=True)
    _use_decision_model(monkeypatch, {"text A": 56})

    async def failing_score_by_requirement(scorer, resume_text, requirements):
        return None
    monkeypatch.setattr("backend.analyzer.decision_scorer.score_resume_by_requirement", failing_score_by_requirement)

    result = await _score("light", {"A": "text A"})

    assert result["scores"] == {"A": 56}


FULL_REQUIREMENTS = {
    "min_years": 5, "education": "Bachelor's in Computer Science", "work_authorization": "US work authorization",
    "must_have": [{"name": "Python", "aliases": ["Py"], "quote": "Python"},
                  {"name": "SQL", "aliases": [], "quote": "SQL"}],
    "nice_to_have": [{"name": "Kafka", "aliases": [], "quote": "Kafka"}],
    "certifications": [{"name": "AWS Solutions Architect", "required": True},
                       {"name": "CKA", "required": False}],
    "responsibilities": ["Build data pipelines"],
}


def test_one_question_is_asked_for_every_requirement():
    from backend.analyzer.decision_scorer import build_requirement_questions

    questions = build_requirement_questions(FULL_REQUIREMENTS)

    assert [question["section"] for question in questions] == [
        "must_have", "must_have", "nice_to_have", "required_certifications", "preferred_certifications",
        "responsibilities", "years", "education", "work_authorization"]
    assert "Python (also called: Py)" in questions[0]["instructions"]


def _jev_answering(monkeypatch, top_level_for):
    """Stand in for Jev: the top level where top_level_for(instructions) is true, the bottom level elsewhere."""
    from backend.analyzer import decision_scorer

    async def fake_ask_model(scorer, api_key, state, questions):
        answers = {key: {"score": len(question["criteria"]) - 1 if top_level_for(question["instructions"]) else 0}
                   for key, question in questions.items()}
        return {"answers": answers, "usage": {"input_tokens": 10, "output_tokens": 1}}
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "key")
    monkeypatch.setattr(decision_scorer, "_ask_model", fake_ask_model)
    return decision_scorer


def _asks_about_a_must_have(instructions):
    return "skill or qualification: Python" in instructions or "skill or qualification: SQL" in instructions


@pytest.mark.asyncio
async def test_per_requirement_points_follow_the_section_weights(monkeypatch):
    decision_scorer = _jev_answering(monkeypatch, _asks_about_a_must_have)

    result = await decision_scorer.score_resume_by_requirement("jev", "resume text", FULL_REQUIREMENTS)

    assert result["score"] == 45
    assert result["sections"]["must_have"]["points"] == 45
    assert result["sections"]["nice_to_have"]["points"] == 0


@pytest.mark.asyncio
async def test_sections_the_posting_lacks_do_not_cost_points(monkeypatch):
    decision_scorer = _jev_answering(monkeypatch, lambda instructions: True)

    result = await decision_scorer.score_resume_by_requirement("jev", "resume text", REQUIREMENTS)

    assert result["score"] == 100
    assert set(result["sections"]) == {"must_have", "nice_to_have", "years"}


TIERED_REQUIREMENTS = {
    "must_have": [{"name": "Swift", "aliases": [], "quote": "Swift", "core": True},
                  {"name": "Agile", "aliases": [], "quote": "Agile", "core": False},
                  {"name": "APIs", "aliases": [], "quote": "APIs", "core": False}],
    "nice_to_have": [{"name": "Kafka", "aliases": [], "quote": "Kafka", "core": False}],
    "certifications": [], "responsibilities": [],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("skills_shown, points_out_of_55", [
    (("Swift",), 30),
    (("Agile", "APIs"), 15),
    (("Kafka",), 10),
])
async def test_core_must_haves_outweigh_other_must_haves_which_outweigh_nice_to_haves(
        monkeypatch, skills_shown, points_out_of_55):
    decision_scorer = _jev_answering(
        monkeypatch, lambda instructions: any(skill in instructions for skill in skills_shown))

    result = await decision_scorer.score_resume_by_requirement("jev", "resume text", TIERED_REQUIREMENTS)

    assert result["score"] == pytest.approx(100 * points_out_of_55 / 55, abs=1)
    assert set(result["sections"]) == {"core_must_have", "must_have", "nice_to_have"}


@pytest.mark.asyncio
async def test_must_haves_keep_their_full_weight_when_all_of_them_are_core(monkeypatch):
    all_core = {**TIERED_REQUIREMENTS,
                "must_have": [{**item, "core": True} for item in TIERED_REQUIREMENTS["must_have"]]}
    decision_scorer = _jev_answering(monkeypatch, lambda instructions: "Kafka" not in instructions)

    result = await decision_scorer.score_resume_by_requirement("jev", "resume text", all_core)

    assert result["sections"]["core_must_have"]["weight"] == 45
    assert result["score"] == round(100 * 45 / 55)


def test_a_spread_of_likely_levels_earns_partial_points():
    from backend.analyzer.decision_scorer import SKILL_LEVELS, _share_earned

    assert _share_earned({"score": 2, "probabilities": {"0": 0.5, "3": 0.5}}, SKILL_LEVELS) == 0.5
    assert _share_earned({"score": 2}, SKILL_LEVELS) == 0.8


@pytest.mark.asyncio
@pytest.mark.parametrize("extracted", [
    None,
    {"must_have": [], "nice_to_have": []},
])
async def test_scoring_reads_the_raw_posting_when_no_requirements_were_extracted(monkeypatch, extracted):
    from backend.analyzer import cv_scorer

    async def fake_requirements_for_job(job_id, job_text):
        return extracted
    monkeypatch.setattr("backend.analyzer.decision_scorer.uses_extracted_requirements", lambda db=None: True)
    monkeypatch.setattr("backend.analyzer.job_requirements.requirements_for_job", fake_requirements_for_job)

    assert await cv_scorer._requirements_for_scoring(FakeJob(), "JD text") is None


@pytest.mark.asyncio
async def test_a_failed_extraction_never_fails_the_score(monkeypatch):
    from backend.analyzer import cv_scorer

    async def failing_requirements_for_job(job_id, job_text):
        raise RuntimeError("claude-code subprocess failed")
    monkeypatch.setattr("backend.analyzer.decision_scorer.uses_extracted_requirements", lambda db=None: True)
    monkeypatch.setattr("backend.analyzer.job_requirements.requirements_for_job", failing_requirements_for_job)

    assert await cv_scorer._requirements_for_scoring(FakeJob(), "JD text") is None


@pytest.mark.asyncio
async def test_the_posting_setting_skips_extraction(monkeypatch):
    from backend.analyzer import cv_scorer

    async def extraction_must_not_run(job_id, job_text):
        raise AssertionError("requirements were extracted with the setting on posting")
    monkeypatch.setattr("backend.analyzer.decision_scorer.uses_extracted_requirements", lambda db=None: False)
    _use_question_set(monkeypatch, per_requirement=False)
    monkeypatch.setattr("backend.analyzer.job_requirements.requirements_for_job", extraction_must_not_run)

    assert await cv_scorer._requirements_for_scoring(FakeJob(), "JD text") is None


def test_the_model_reads_requirements_in_place_of_the_posting():
    from backend.analyzer.decision_scorer import _build_state

    with_requirements = _build_state("resume text", "the raw posting text", REQUIREMENTS)
    assert "Must-have requirements: Python (also: Py)" in with_requirements
    assert "Nice-to-have requirements: Kafka" in with_requirements
    assert "Minimum years of experience: 5" in with_requirements
    assert "the raw posting text" not in with_requirements

    without_requirements = _build_state("resume text", "the raw posting text")
    assert "the raw posting text" in without_requirements


@pytest.mark.parametrize("stored_value, expected", [
    (None, "jev"),
    ("jev", "jev"),
    ("glide", "glide"),
    ("llm", "llm"),
    ("unknown", "llm"),
])
def test_ats_scorer_reads_the_setting(monkeypatch, stored_value, expected):
    from backend.models.db import Setting
    from backend.analyzer import decision_scorer

    engine = create_engine("sqlite:///:memory:")
    Setting.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    if stored_value is not None:
        session = Session()
        session.add(Setting(key="ats_scorer", value=stored_value))
        session.commit()
        session.close()
    monkeypatch.setattr(decision_scorer, "SessionLocal", Session)

    assert real_ats_scorer() == expected


@pytest.mark.asyncio
async def test_the_score_is_the_weighted_sum_of_the_graded_answers(monkeypatch):
    from backend.analyzer import decision_scorer

    levels_picked = {"core_match": 2, "skill_match": 3, "responsibility_match": 2, "experience_match": 3,
                     "requirements_match": 3, "domain_match": 1}

    async def fake_ask_model(scorer, api_key, state, questions=None):
        answers = {key: {"score": level} for key, level in levels_picked.items()}
        answers.update({key: {"noul": 0.0} for key in decision_scorer.TRUE_FALSE_QUESTIONS})
        return {"answers": answers, "usage": {}}
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "key")
    monkeypatch.setattr(decision_scorer, "_ask_model", fake_ask_model)

    result = await decision_scorer.score_resume("jev", "resume text", "job text")

    highest_level = 3
    expected = 100 * sum(weight * levels_picked[key] / highest_level
                         for key, weight in decision_scorer.QUESTION_WEIGHTS.items())
    assert result["score"] == round(expected)
    assert result["breakdown"]["domain"] == 7


def test_the_question_weights_add_up_to_the_whole_score():
    from backend.analyzer.decision_scorer import GRADED_QUESTIONS, QUESTION_WEIGHTS

    assert sum(QUESTION_WEIGHTS.values()) == pytest.approx(1.0)
    assert set(QUESTION_WEIGHTS) <= set(GRADED_QUESTIONS)


@pytest.mark.asyncio
async def test_the_sections_setting_asks_each_question_with_its_own_part_of_the_posting(scorer_db, monkeypatch):
    _use_posting_sections(monkeypatch, {"must_have": ["Python"]})
    _use_decision_model(monkeypatch, {})
    sections_seen = []

    async def fake_score_by_section(scorer, resume_text, job_text, sections):
        sections_seen.append(sections)
        return _decision_model_result(71)
    monkeypatch.setattr("backend.analyzer.decision_scorer.score_resume_by_posting_section", fake_score_by_section)

    result = await _score("light", {"A": "text A"})

    assert result["scores"] == {"A": 71}
    assert sections_seen == [{"must_have": ["Python"]}]


@pytest.mark.asyncio
async def test_a_posting_that_cannot_be_split_is_scored_whole(monkeypatch):
    from backend.analyzer import cv_scorer

    def failing_split_posting(job_text):
        raise RuntimeError("could not read the posting")
    monkeypatch.setattr("backend.analyzer.decision_scorer.uses_posting_sections", lambda db=None: True)
    monkeypatch.setattr("backend.analyzer.job_sections.split_posting", failing_split_posting)

    assert await cv_scorer._sections_for_scoring(FakeJob(), "JD text") is None


POSTING_WITH_SECTIONS = """About us
Acme builds payment software for banks across Europe and has done so for twenty years.

Responsibilities
- Build and run the data pipelines that feed the fraud models every night
- Review code and mentor two junior engineers on the team
- Own the on-call rota for the ingestion services and write the incident reviews
- Work with the risk analysts to turn their questions into reliable datasets

Requirements
- 5+ years of experience with Python and SQL in production systems
- Bachelor's degree in Computer Science or a related field
- Experience with Kafka is a plus

Benefits
- Free lunch and a gym membership for every employee
"""


def test_each_question_reads_only_its_own_part_of_the_posting():
    from backend.analyzer.decision_scorer import job_context_for_question
    from backend.analyzer.job_sections import split_posting

    sections = split_posting(POSTING_WITH_SECTIONS)
    duties = job_context_for_question("responsibility_match", sections, POSTING_WITH_SECTIONS)
    skills = job_context_for_question("skill_match", sections, POSTING_WITH_SECTIONS)

    assert "fraud models" in duties and "Bachelor" not in duties
    assert "Python and SQL" in skills and "fraud models" not in skills
    assert "Free lunch" not in duties + skills


def test_a_question_reads_the_whole_posting_only_when_it_has_no_part_of_its_own():
    from backend.analyzer.decision_scorer import job_context_for_question
    from backend.analyzer.job_sections import split_posting

    posting = "We need someone good with data. Join us and grow with the team."

    assert job_context_for_question("responsibility_match", split_posting(posting), posting) == posting


@pytest.mark.asyncio
async def test_a_jev_score_carries_the_matched_and_missing_keywords(scorer_db, monkeypatch):
    monkeypatch.setattr("backend.analyzer.cv_scorer.call_llm", _llm_returning("{}", []))
    _use_decision_model(monkeypatch, {"Built pipelines in Python": 56})
    from backend.analyzer import cv_scorer

    result = await cv_scorer.score_job_sync(FakeJob(), {"A": "Built pipelines in Python"}, db=None, depth="light",
                                            preloaded_text="We need Python and Kubernetes.")

    report = result["_scoring_report"]
    assert report["matched_keywords"] == ["Python"]
    assert report["missing_keywords"] == ["Kubernetes"]
    assert report["scored_by"] == "jev"
