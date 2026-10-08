"""The model only classifies; the score is worked out in code from its picks."""
import json

import pytest

from backend.analyzer import llm_ats_score

BEST_PICKS = {"role_fit": "same_work", "skills": "all", "responsibilities": "all", "experience": "meets",
              "qualifications": "all"}


def _with_evidence(picks):
    """The reply the model gives: for each question, its evidence and then its pick."""
    return {question: {"matched": [f"{question} item: shown in the resume"], "missing": [f"{question} gap"],
                       "reason": f"Why {category} fits.", "category": category}
            for question, category in picks.items()}


def _model_replies(monkeypatch, picks, input_tokens=5000, output_tokens=40):
    calls = []

    async def fake_call_llm(prompt, system, **options):
        calls.append({"prompt": prompt, "system": system, **options})
        text = picks if isinstance(picks, str) else json.dumps(_with_evidence(picks))
        return {"text": text, "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}
    monkeypatch.setattr(llm_ats_score, "call_llm", fake_call_llm)
    return calls


def test_the_best_pick_for_every_question_scores_100():
    assert llm_ats_score.score_from_picks(BEST_PICKS)["score"] == 100


def test_the_parts_are_added_by_weight_and_scaled_by_role_fit():
    picks = {"role_fit": "same_work_with_gaps", "skills": "most", "responsibilities": "about_half",
             "experience": "meets", "qualifications": "some"}

    result = llm_ats_score.score_from_picks(picks)

    # Parts: 0.45 x 0.75 + 0.25 x 0.5 + 0.15 x 1.0 + 0.15 x 0.5 = 0.69. Score: 0.75 x (0.4 + 0.6 x 0.69).
    assert result["total_of_parts"] == 0.69
    assert result["score"] == round(100 * 0.75 * (0.4 + 0.6 * 0.69))


def test_a_part_the_posting_does_not_state_is_left_out():
    picks = {**BEST_PICKS, "skills": "about_half", "responsibilities": "not_stated", "qualifications": "not_stated"}

    result = llm_ats_score.score_from_picks(picks)

    # Only skills (0.45) and experience (0.15) count: (0.45 x 0.5 + 0.15 x 1.0) / 0.60
    assert result["parts"]["responsibilities"] is None
    assert result["total_of_parts"] == 0.62


def test_a_resume_of_a_different_kind_of_work_scores_zero_whatever_else_it_covers():
    assert llm_ats_score.score_from_picks({**BEST_PICKS, "role_fit": "different_work"})["score"] == 0


def test_the_reply_must_be_strict_json_with_one_category_per_question():
    schema = llm_ats_score._schema_of_the_reply()

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["role_fit", "skills", "responsibilities", "experience", "qualifications"]
    experience = schema["properties"]["experience"]
    assert experience["required"] == ["matched", "missing", "reason", "category"]
    assert experience["additionalProperties"] is False
    assert experience["properties"]["category"]["enum"] == ["far_below", "below", "meets", "not_stated"]
    assert experience["properties"]["matched"] == {"type": "array", "items": {"type": "string"}}


def test_the_cost_uses_the_price_per_million_tokens():
    # 10,000 in at $0.035 per million, 100 out at $0.60 per million.
    assert llm_ats_score.cost_in_usd(10_000, 100) == round(0.00035 + 0.00006, 6)


@pytest.mark.asyncio
async def test_the_score_comes_back_with_the_picks_and_the_cost(monkeypatch):
    calls = _model_replies(monkeypatch, BEST_PICKS, input_tokens=6000, output_tokens=50)

    result = await llm_ats_score.ats_score("Ran Kubernetes clusters.", "Requirements\n- Kubernetes")

    assert result["score"] == 100
    assert result["picks"] == BEST_PICKS
    assert result["evidence"]["skills"] == {"matched": ["skills item: shown in the resume"], "missing": ["skills gap"],
                                            "reason": "Why all fits."}
    assert (result["input_tokens"], result["output_tokens"]) == (6000, 50)
    assert result["cost_usd"] == llm_ats_score.cost_in_usd(6000, 50)
    assert calls[0]["model"] == "deepseek/deepseek-v4.1-flash"
    assert calls[0]["response_schema"] == llm_ats_score._schema_of_the_reply()
    assert "Ran Kubernetes clusters." in calls[0]["prompt"] and "- Kubernetes" in calls[0]["prompt"]


@pytest.mark.asyncio
async def test_a_reply_with_an_unknown_category_gives_nothing(monkeypatch):
    _model_replies(monkeypatch, {**BEST_PICKS, "skills": "plenty"})

    assert await llm_ats_score.ats_score("resume", "posting") is None


@pytest.mark.asyncio
async def test_a_reply_that_is_not_json_gives_nothing(monkeypatch):
    _model_replies(monkeypatch, "The candidate looks like a good match.")

    assert await llm_ats_score.ats_score("resume", "posting") is None


@pytest.mark.asyncio
async def test_a_reply_with_a_bare_category_and_no_evidence_gives_nothing(monkeypatch):
    _model_replies(monkeypatch, json.dumps(BEST_PICKS))

    assert await llm_ats_score.ats_score("resume", "posting") is None
