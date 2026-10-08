"""The match score with a decision model judging each keyword's importance to the job and evidence in the résumé."""
import pytest

from backend.analyzer import decision_scorer, judged_match

POSTING = "iOS Engineer. You will build our app in Swift. Jira is used for tickets."
RESUME = "Skills: Swift, Jira.\nBuilt dashboards and tracked work in Jira every day."
IMPORTANCE = {"Swift": 2, "Jira": 0}          # core, optional
EVIDENCE = {"Swift": 1, "Jira": 4}            # only listed, main work


states_seen = []


def _jev_answering(monkeypatch, importance=IMPORTANCE, evidence=EVIDENCE):
    states_seen.clear()

    async def fake_ask_model(scorer, api_key, state, questions):
        levels = evidence if "CANDIDATE RESUME" in state else importance
        states_seen.append(state)
        answers = {key: {"score": next(level for keyword, level in levels.items() if keyword in question["instructions"])}
                   for key, question in questions.items()}
        return {"answers": answers, "usage": {}}
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "key")
    monkeypatch.setattr(decision_scorer, "_ask_model", fake_ask_model)


@pytest.mark.asyncio
async def test_a_core_tool_only_listed_earns_little_however_well_the_optional_ones_are_shown(monkeypatch):
    _jev_answering(monkeypatch)

    result = await judged_match.judged_match_score("jev", RESUME, POSTING)

    # Swift: weight 4, credit 0.3. Jira: weight 1, credit 1.0. Hard skills are the posting's only part,
    # so they hold all 100 points and earn (1.2 + 1.0) / 5 of them.
    assert result["match_breakdown"] == {"hard": 44.0}
    assert result["match_score_by_count"] > result["match_score"]


@pytest.mark.asyncio
async def test_each_keyword_row_says_how_it_was_judged(monkeypatch):
    _jev_answering(monkeypatch)

    table = (await judged_match.judged_match_score("jev", RESUME, POSTING))["keyword_table"]

    assert table[0] == {"keyword": "Swift", "type": "hard", "importance": "core", "evidence": "only listed",
                        "weight": 4.0, "credit": 0.3}
    assert table[1]["importance"] == "optional" and table[1]["evidence"] == "main work"


@pytest.mark.asyncio
async def test_the_title_is_scored_by_rule_and_a_degree_nobody_asked_for_is_left_out(monkeypatch):
    _jev_answering(monkeypatch)

    result = await judged_match.judged_match_score("jev", "iOS Engineer\n" + RESUME, POSTING, "iOS Engineer")

    # Swift is the one must-have and is only listed, so 30% of it is met and the title earns 30% of its 14.3.
    assert result["match_out_of"]["title"] == 14.3
    assert result["match_breakdown"]["title"] == 4.3
    assert "degree" not in result["match_breakdown"]


@pytest.mark.asyncio
async def test_an_optional_hard_skill_does_not_hold_the_other_parts_back(monkeypatch):
    _jev_answering(monkeypatch, evidence={"Swift": 4, "Jira": 0})

    result = await judged_match.judged_match_score("jev", "iOS Engineer\n" + RESUME, POSTING, "iOS Engineer")

    # Swift, the must-have, is fully shown; Jira is optional and missing. The title keeps all its points.
    assert result["match_details"]["must_have_hard_skills_met"] == 1.0
    assert result["match_breakdown"]["title"] == result["match_out_of"]["title"]


@pytest.mark.asyncio
async def test_nothing_is_returned_when_the_model_cannot_be_asked(monkeypatch):
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "")

    assert await judged_match.judged_match_score("jev", RESUME, POSTING) is None


@pytest.mark.asyncio
async def test_nothing_is_returned_when_the_model_fails(monkeypatch):
    async def failing_ask_model(scorer, api_key, state, questions):
        raise RuntimeError("server error")
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "key")
    monkeypatch.setattr(decision_scorer, "_ask_model", failing_ask_model)

    assert await judged_match.judged_match_score("jev", RESUME, POSTING) is None


@pytest.mark.asyncio
async def test_evidence_is_judged_with_the_posting_and_the_must_haves_in_view(monkeypatch):
    _jev_answering(monkeypatch)

    await judged_match.judged_match_score("jev", RESUME, POSTING)

    importance_state, evidence_state = states_seen
    assert "CANDIDATE RESUME" not in importance_state
    assert "Built dashboards" in evidence_state and "build our app in Swift" in evidence_state
    # Swift was judged core, Jira optional: only Swift is handed over as a must-have.
    must_haves = evidence_state.split("<<<MUST-HAVE HARD SKILLS FOR THIS JOB>>>\n")[-1]
    assert "- Swift" in must_haves and "Jira" not in must_haves
