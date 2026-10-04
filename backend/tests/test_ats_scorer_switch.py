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
    return Session


def _use_decision_model(monkeypatch, scores_by_resume_text):
    async def fake_score_resume(scorer, resume_text, job_text):
        score = scores_by_resume_text.get(resume_text)
        return None if score is None else _decision_model_result(score)
    monkeypatch.setattr("backend.analyzer.decision_scorer.score_resume", fake_score_resume)


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
