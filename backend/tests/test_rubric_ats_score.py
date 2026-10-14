"""The rubric ATS score: one model call and a keyword check, with the arithmetic done in code."""
import json

import pytest

from backend.analyzer import rubric_ats_score as rubric

POSTING = "Senior DevOps Engineer\nRequirements\n- Hands-on Kubernetes and Terraform\n- Experience with Datadog"
RESUME = ("Jordan Lee\nDevOps Engineer\n## Summary\nDevOps engineer.\n## Experience\nDevOps Engineer at Acme\n"
          "- Ran Kubernetes clusters\n## Skills\nKubernetes, Terraform, Linux")
RESUME_WITH_A_FIGURE = RESUME.replace("- Ran Kubernetes clusters", "- Ran Kubernetes clusters for 40 teams")
BEST_PICKS = {"occupation": "same", "title": "exact", "core_skills": "all", "other_skills": "all",
              "experience": "meets", "qualifications": "all", "evidence": "strong"}


def _score(picks, keyword_share, core_skills_used_share=None):
    """The score with every duty met, so a test can look at one other part at a time."""
    return rubric.score_from_picks(picks, keyword_share, core_skills_used_share, duties_share=1.0)


@pytest.fixture(autouse=True)
def nothing_remembered():
    rubric._scores_remembered.clear()


def _reply(picks=BEST_PICKS, missing=("Datadog",), core=("Kubernetes",), met=("Run clusters",), not_met=(),
           another_way=()):
    answers = {question: {"reason": f"why {question}", "option": option} for question, option in picks.items()}
    shown = [{"keyword": keyword, "shown_by": shown_by} for keyword, shown_by in another_way]
    return {"text": json.dumps({"core_skills_named": list(core), "duties_met": list(met), "duties_not_met": list(not_met),
                                **answers, "shown_another_way": shown, "missing": list(missing)}),
            "usage": {"input_tokens": 5000, "output_tokens": 300},
            "model": "google/gemma-4-31b-it"}


def _model_replies(monkeypatch, *replies):
    """The model gives these replies in turn, and the last one again for every call after that."""
    calls = []

    async def fake_call_llm(prompt, system, **options):
        calls.append({"prompt": prompt, "system": system, **options})
        reply = replies[min(len(calls), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(rubric, "call_llm", fake_call_llm)
    return calls


# ── The keyword match ──────────────────────────────────────────────────────────

def test_keywords_are_sorted_by_where_the_resume_names_them():
    found = rubric.keyword_check(RESUME, POSTING)

    assert found["used"] == ["Kubernetes"]
    assert found["listed"] == ["Terraform"]
    assert found["missing"] == ["Datadog"]
    assert found["alternatives_not_needed"] == []
    assert found["shown_another_way"] == []
    assert found["share"] == round((1 + 0.4) / 3, 2)


# ── A choice of keywords is one requirement, met by any one of them ────────────

PYTHON_RESUME = "Jordan\n## Experience\nEngineer at Acme\n- Wrote Python services on AWS\n## Skills\nGrafana"


def test_one_keyword_of_a_choice_meets_the_whole_requirement():
    posting = "Requirements\n- Proficiency in at least one language (e.g. Clojure, TypeScript, Python)\n- AWS"

    found = rubric.keyword_check(PYTHON_RESUME, posting)

    assert found["used"] == ["AWS", "Python"]
    assert found["missing"] == []
    assert sorted(found["alternatives_not_needed"]) == ["Clojure", "TypeScript"]
    assert found["share"] == 1.0


def test_an_or_between_two_keywords_is_a_choice():
    assert rubric.keyword_check(PYTHON_RESUME, "Requirements\n- Experience with AWS or Azure")["share"] == 1.0


def test_a_choice_none_of_whose_keywords_the_resume_names_is_one_missing_requirement():
    found = rubric.keyword_check(PYTHON_RESUME, "Requirements\n- Terraform or Pulumi\n- AWS")

    assert found["missing"] == ["Terraform or Pulumi"]
    assert found["share"] == 0.5


def test_a_choice_is_judged_by_its_best_keyword():
    found = rubric.keyword_check(PYTHON_RESUME, "Requirements\n- Monitoring tools such as Datadog, Grafana or Prometheus")

    assert found["listed"] == ["Grafana"]           # only in the skills list, so the choice earns the listed share
    assert found["share"] == 0.4


def test_keywords_joined_by_and_are_each_required():
    found = rubric.keyword_check(PYTHON_RESUME, "Requirements\n- Hands-on Kubernetes, Docker and AWS")

    assert sorted(found["missing"]) == ["Docker", "Kubernetes"]
    assert found["share"] == round(1 / 3, 2)


def test_an_or_that_is_not_between_the_keywords_offers_no_choice():
    found = rubric.keyword_check(PYTHON_RESUME, "Requirements\n- Build applications or services using AWS and Docker")

    assert found["missing"] == ["Docker"]


def test_a_keyword_also_asked_for_alone_is_required_whatever_a_choice_says():
    found = rubric.keyword_check(PYTHON_RESUME, "Requirements\n- Python or TypeScript\n- Strong TypeScript skills")

    assert found["missing"] == ["TypeScript"]
    assert found["used"] == ["Python"]


def test_a_posting_that_names_no_keyword_leaves_the_keyword_part_out():
    assert rubric.keyword_check(RESUME, "We want a kind and careful person.")["share"] is None


# ── The arithmetic ─────────────────────────────────────────────────────────────

def test_the_best_picks_and_every_keyword_used_score_one_hundred():
    assert _score(BEST_PICKS, 1.0)["ats_score"] == 100


def test_the_weights_of_the_parts_add_up_to_one():
    assert sum(rubric.WEIGHT_OF_PART.values()) == pytest.approx(1.0)
    assert set(rubric.WEIGHT_OF_PART) == (set(rubric.RUBRIC) - {"occupation"}) | {"keywords", "duties"}


def test_a_different_occupation_keeps_a_tenth_of_the_score():
    assert _score({**BEST_PICKS, "occupation": "different"}, 1.0)["ats_score"] == 10


def test_few_core_skills_scale_the_whole_score_as_well_as_their_own_part():
    result = _score({**BEST_PICKS, "core_skills": "few"}, 1.0)

    assert result["core_skills_kept"] == 0.6
    assert result["ats_score"] == round(100 * (1 - 0.30 * 0.75) * 0.6)


def test_the_core_skills_pick_can_be_no_higher_than_the_share_the_roles_name():
    """The model said "all", but the roles name only half of the skills it called core."""
    result = _score(BEST_PICKS, 1.0, core_skills_used_share=0.5)

    assert result["parts"]["core_skills"] == 0.5
    assert result["core_skills_kept"] == 0.85
    assert result["ats_score"] == round(100 * (1 - 0.30 * 0.5) * 0.85)


def test_the_code_never_raises_the_models_core_skills_pick():
    result = _score({**BEST_PICKS, "core_skills": "few"}, 1.0, core_skills_used_share=1.0)

    assert result["parts"]["core_skills"] == 0.25


def test_core_skills_are_looked_for_in_the_roles_and_not_in_the_skills_list():
    check = rubric.core_skill_check(RESUME, ["K8s", "Terraform", "Karpenter"])

    assert check["used"] == ["K8s"]                       # the roles say Kubernetes, which is the same skill
    assert check["not_used"] == ["Terraform", "Karpenter"]   # Terraform is only in the skills list
    assert check["share"] == round(1 / 3, 2)


def test_the_share_kept_rises_evenly_between_its_points():
    assert rubric._share_kept_for_core_skills(0.0) == 0.4
    assert rubric._share_kept_for_core_skills(0.65) == pytest.approx(0.925)
    assert rubric._share_kept_for_core_skills(0.9) == 1.0


def test_a_part_the_posting_does_not_ask_for_gives_its_weight_to_the_others():
    result = _score({**BEST_PICKS, "experience": "not_stated", "qualifications": "not_stated"}, None)

    assert result["parts"]["experience"] is None
    assert result["ats_score"] == 100


# ── What the model is sent, and what comes back ────────────────────────────────

def test_the_model_is_not_sent_the_skills_list_and_is_told_what_the_keyword_check_found():
    prompt = rubric._prompt(RESUME, POSTING, rubric.keyword_check(RESUME, POSTING))

    assert "Ran Kubernetes clusters" in prompt
    assert "Kubernetes, Terraform, Linux" not in prompt
    assert "Only listed as skills: Terraform" in prompt
    assert "Not in the resume: Datadog" in prompt


def test_every_question_asks_for_a_reason_before_one_of_its_own_options():
    schema = rubric._schema_of_the_reply()

    for question, details in rubric.RUBRIC.items():
        assert list(schema["properties"][question]["properties"]) == ["reason", "option"]
        assert schema["properties"][question]["properties"]["option"]["enum"] == list(details["options"])
    assert schema["required"] == (["core_skills_named", "duties_met", "duties_not_met"] + list(rubric.RUBRIC)
                                  + ["shown_another_way", "missing"])


def test_the_prompts_stay_small():
    rubric_part = rubric._prompt("", "", {"used": [], "listed": [], "missing": []})

    assert len(rubric.SYSTEM_PROMPT) < 1400
    assert len(rubric_part) < 3000


@pytest.mark.asyncio
async def test_three_calls_give_the_score_the_reasons_and_the_cost(monkeypatch):
    calls = _model_replies(monkeypatch, _reply())

    result = await rubric.ats_score(RESUME_WITH_A_FIGURE, POSTING)

    assert len(calls) == rubric.TIMES_ASKED == 3
    assert (calls[0]["provider"], calls[0]["model"]) == ("openrouter", "google/gemma-4-31b-it")
    assert calls[0]["temperature"] == 0
    assert result["scorer"] == "rubric"
    assert result["ats_score"] == round(100 * (0.85 + 0.15 * 0.47))
    assert result["missing"] == ["Datadog"]
    assert result["core_skills"] == {"named": ["Kubernetes"], "used": ["Kubernetes"], "not_used": [], "share": 1.0}
    row = result["rubric"][0]
    assert (row["question"], row["option"], row["reason"]) == ("occupation", "same", "why occupation")
    assert (result["input_tokens"], result["output_tokens"]) == (15000, 900)


@pytest.mark.asyncio
async def test_a_reply_with_an_option_that_is_not_the_questions_gives_no_score(monkeypatch):
    _model_replies(monkeypatch, _reply({**BEST_PICKS, "title": "perfect"}))

    assert await rubric.ats_score(RESUME, POSTING) is None


@pytest.mark.asyncio
async def test_a_failed_call_or_a_reply_that_is_not_json_gives_no_score(monkeypatch):
    _model_replies(monkeypatch, RuntimeError("no credit"))
    assert await rubric.ats_score(RESUME, POSTING) is None

    _model_replies(monkeypatch, {"text": "I cannot do that.", "usage": {}})
    assert await rubric.ats_score(RESUME, POSTING) is None


def test_the_models_a_caller_may_name():
    assert rubric.MODELS == {"gemma": ("openrouter", "google/gemma-4-31b-it"),
                             "haiku": ("openrouter", "anthropic/claude-haiku-5.5"),
                             "luna": ("openrouter", "openai/gpt-6-luna"),
                             "haiku_plan": ("claude_code", "claude-haiku-5-5"),
                             "luna_plan": ("codex_cli", "gpt-6-luna"),
                             "luna_plan_high": ("codex_cli", "gpt-6-luna:high")}


# ── What keeps the score steady ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_middle_pick_of_the_three_replies_is_the_one_used(monkeypatch):
    _model_replies(monkeypatch, _reply({**BEST_PICKS, "other_skills": "all"}), _reply({**BEST_PICKS, "other_skills": "few"}),
                   _reply({**BEST_PICKS, "other_skills": "most"}))

    result = await rubric.ats_score(RESUME, POSTING)

    other = [row for row in result["rubric"] if row["question"] == "other_skills"][0]
    assert other["option"] == "most"
    assert other["option_share"] == 0.8
    assert other["share"] == round((1.0 + 0.25 + 0.8) / 3, 2)      # it earns the average of the three picks
    assert result["times_asked"] == 3


@pytest.mark.asyncio
async def test_a_question_the_model_is_in_two_minds_about_moves_the_score_only_a_little(monkeypatch):
    """Other skills flips between all and most: one flip in three moves the score by a third of that step."""
    _model_replies(monkeypatch, _reply())
    all_three = (await rubric.ats_score(RESUME, POSTING))["ats_score"]
    rubric._scores_remembered.clear()
    _model_replies(monkeypatch, _reply({**BEST_PICKS, "other_skills": "most"}), _reply())
    one_flip = (await rubric.ats_score(RESUME, POSTING))["ats_score"]

    assert 0 <= all_three - one_flip <= 2


def test_a_question_most_replies_call_not_asked_is_left_out():
    def reply(experience):
        return {question: {"option": experience if question == "experience" else option}
                for question, option in BEST_PICKS.items()}

    assert rubric._average_shares([reply("not_stated"), reply("not_stated"), reply("meets")])["experience"] is None
    assert rubric._average_shares([reply("not_stated"), reply("slightly_below"), reply("meets")])["experience"] == 0.8


@pytest.mark.asyncio
async def test_a_reply_that_fails_is_left_out_and_the_others_still_score(monkeypatch):
    _model_replies(monkeypatch, RuntimeError("busy"), _reply())

    result = await rubric.ats_score(RESUME, POSTING)

    assert result["times_asked"] == 2


@pytest.mark.asyncio
async def test_the_same_resume_and_posting_are_scored_once_and_remembered(monkeypatch):
    calls = _model_replies(monkeypatch, _reply())

    first = await rubric.ats_score(RESUME, POSTING)
    second = await rubric.ats_score(RESUME, POSTING)

    assert (first["remembered"], second["remembered"]) == (False, True)
    assert {**second, "remembered": False} == first
    assert len(calls) == 3
    await rubric.ats_score(RESUME + " ", POSTING)      # a different résumé is asked about afresh
    assert len(calls) == 6


def test_core_skills_count_when_most_replies_name_them():
    replies = [{"core_skills_named": ["Kubernetes", "AWS"]}, {"core_skills_named": ["kubernetes", "Docker"]},
               {"core_skills_named": ["Kubernetes", "AWS", "Terraform"]}]

    assert sorted(rubric._core_skills_most_replies_name(replies), key=str.lower) == ["AWS", "Kubernetes"]


# ── The years of work, from the dates ──────────────────────────────────────────

from datetime import date

DATED = ("Jordan\n## Experience\nSenior Engineer at Acme (Jul 2023 – Present\n- Ran clusters in 2019 – 2020 style\n"
         "Engineer at Beta (Jan 2020 – Jun 2023\nIntern at Beta (Jan 2020 – Jun 2020\n## Skills\nLinux")


def test_years_are_added_up_from_the_roles_and_overlaps_count_once():
    assert rubric.years_of_work(DATED, today=date(2026, 6, 30)) == 6.5      # Jan 2020 to Jun 2026


def test_a_resume_with_no_dated_role_gives_no_years():
    assert rubric.years_of_work(RESUME) is None


def test_the_years_a_posting_asks_for_are_its_largest_minimum():
    assert rubric.years_the_posting_asks_for("5+ years of DevOps. 2 years of Go. Range 10-15 Years overall.") == 10
    assert rubric.years_the_posting_asks_for("Proven experience in DevOps.") is None


def test_the_dates_settle_the_experience_question_when_the_posting_states_years():
    assert rubric.experience_from_the_dates(DATED, "8+ years of experience")["option"] in ("slightly_below", "far_below")
    assert rubric.experience_from_the_dates(DATED, "3+ years of experience")["option"] == "meets"
    assert rubric.experience_from_the_dates(DATED, "Proven experience") is None


@pytest.mark.asyncio
async def test_code_overrules_the_models_experience_pick_when_the_dates_say_otherwise(monkeypatch):
    _model_replies(monkeypatch, _reply({**BEST_PICKS, "experience": "meets"}))

    result = await rubric.ats_score(DATED, POSTING + "\n- 15+ years of experience")

    experience = [row for row in result["rubric"] if row["question"] == "experience"][0]
    assert experience["option"] == "far_below"
    assert "Counted from the dates" in experience["reason"]


def test_between_all_and_most_the_count_from_the_roles_decides_the_core_skills():
    assert rubric._core_skills_share("all", 1.0) == rubric._core_skills_share("most", 1.0) == 1.0
    assert rubric._core_skills_share("all", 0.5) == 0.5


def test_a_pick_of_half_or_less_can_only_lower_the_core_skills():
    assert rubric._core_skills_share("about_half", 1.0) == 0.5
    assert rubric._core_skills_share("few", 0.1) == 0.1
    assert rubric._core_skills_share("few", None) == 0.25


@pytest.mark.asyncio
async def test_one_odd_reply_about_the_occupation_does_not_move_the_score(monkeypatch):
    _model_replies(monkeypatch, _reply())
    steady = (await rubric.ats_score(RESUME, POSTING))["ats_score"]
    rubric._scores_remembered.clear()
    _model_replies(monkeypatch, _reply({**BEST_PICKS, "occupation": "different"}), _reply())

    assert (await rubric.ats_score(RESUME, POSTING))["ats_score"] == steady


# ── The duties are counted, not estimated ──────────────────────────────────────

@pytest.mark.asyncio
async def test_the_duties_earn_the_share_the_model_listed_as_met(monkeypatch):
    _model_replies(monkeypatch, _reply(met=("Run clusters", "Build pipelines", "Monitor production"),
                                       not_met=("Disaster recovery",)))

    result = await rubric.ats_score(RESUME, POSTING)

    duties = [row for row in result["rubric"] if row["question"] == "duties"][0]
    assert duties["meaning"] == "3 of 4 duties"
    assert duties["share"] == 0.75
    assert duties["reason"] == "Not shown in a role: Disaster recovery"
    assert [row["question"] for row in result["rubric"]][:5] == ["occupation", "title", "core_skills", "other_skills", "duties"]


@pytest.mark.asyncio
async def test_the_duties_average_over_the_replies(monkeypatch):
    _model_replies(monkeypatch, _reply(met=("a", "b"), not_met=()), _reply(met=("a",), not_met=("b",)),
                   _reply(met=("a", "b"), not_met=()))

    assert (await rubric.ats_score(RESUME, POSTING))["duties"]["share"] == round((1 + 0.5 + 1) / 3, 2)


@pytest.mark.asyncio
async def test_a_posting_with_no_duties_leaves_the_duties_out(monkeypatch):
    _model_replies(monkeypatch, _reply(met=(), not_met=()))

    result = await rubric.ats_score(RESUME, POSTING)

    assert result["parts"]["duties"] is None
    assert [row for row in result["rubric"] if row["question"] == "duties"][0]["meaning"] == "The posting describes no duties"


# ── Strong evidence needs a figure in the roles ────────────────────────────────

def test_a_percentage_an_amount_a_multiple_or_a_count_in_a_bullet_is_a_figure():
    def role(bullet):
        return f"Jordan\n## Experience\nEngineer at Acme (2020 – 2024\n- {bullet}\n## Skills\nLinux"

    for bullet in ("Cut build time by 30%", "Saved $2M a year", "Made deploys 3x faster", "Supported 40 teams",
                   "Ran 1,200 servers"):
        assert rubric.roles_state_a_result_in_numbers(role(bullet)), bullet


def test_a_version_a_product_name_or_a_year_is_not_a_figure():
    def role(bullet):
        return f"Jordan\n## Experience\nEngineer at Acme (2020 – 2024\n- {bullet}\n## Skills\n99% uptime"

    for bullet in ("Built services on Java 17 and Spring Boot 3", "Managed Route 53 records and S3 buckets",
                   "Kept SOC 2 evidence", "Reduced deployment failures and lowered MTTR", "Started in 2019 with the team"):
        assert not rubric.roles_state_a_result_in_numbers(role(bullet)), bullet


@pytest.mark.asyncio
async def test_evidence_the_model_calls_strong_is_held_at_adequate_without_a_figure(monkeypatch):
    _model_replies(monkeypatch, _reply())

    without_a_figure = await rubric.ats_score(RESUME, POSTING)
    with_a_figure = await rubric.ats_score(RESUME_WITH_A_FIGURE, POSTING)

    held = [row for row in without_a_figure["rubric"] if row["question"] == "evidence"][0]
    assert (held["option"], held["share"]) == ("adequate", 0.7)
    assert "No figure" in held["reason"]
    kept = [row for row in with_a_figure["rubric"] if row["question"] == "evidence"][0]
    assert (kept["option"], kept["share"]) == ("strong", 1.0)


# ── A keyword a role shows under another name ──────────────────────────────────

RESUME_WITH_OPENTOFU = RESUME.replace("- Ran Kubernetes clusters", "- Ran Kubernetes clusters\n- Provisioned networks with OpenTofu")

@pytest.mark.asyncio
async def test_a_keyword_most_replies_say_a_role_shows_another_way_counts_as_used(monkeypatch):
    """Terraform is only in the skills list of this resume; a role shows it as OpenTofu."""
    _model_replies(monkeypatch, _reply(another_way=[("Terraform", "OpenTofu")]), _reply(another_way=[("terraform", "OpenTofu")]),
                   _reply())

    keywords = (await rubric.ats_score(RESUME_WITH_OPENTOFU, POSTING))["keywords"]

    assert "Terraform" in keywords["used"] and "Terraform" not in keywords["listed"]
    assert keywords["shown_another_way"] == [{"keyword": "Terraform", "shown_by": "OpenTofu"}]
    assert keywords["share"] == round(2 / 3, 2)


@pytest.mark.asyncio
async def test_one_reply_alone_does_not_move_a_keyword(monkeypatch):
    _model_replies(monkeypatch, _reply(another_way=[("Datadog", "CloudWatch")]), _reply())

    keywords = (await rubric.ats_score(RESUME, POSTING))["keywords"]

    assert keywords["missing"] == ["Datadog"]
    assert keywords["shown_another_way"] == []


@pytest.mark.asyncio
async def test_a_keyword_the_check_never_listed_cannot_be_added(monkeypatch):
    _model_replies(monkeypatch, _reply(another_way=[("Snowflake", "BigQuery")]))

    keywords = (await rubric.ats_score(RESUME, POSTING))["keywords"]

    assert keywords["shown_another_way"] == []
    assert keywords["share"] == round((1 + 0.4) / 3, 2)


def test_a_core_skill_shown_under_another_name_counts_as_named_in_the_roles():
    check = rubric.core_skill_check(RESUME, ["Kubernetes", "Terraform"], [{"keyword": "Terraform", "shown_by": "OpenTofu"}])

    assert check["used"] == ["Kubernetes", "Terraform"]
    assert check["share"] == 1.0


# ── The model cannot give a reason the résumé does not contain ─────────────────

@pytest.mark.asyncio
async def test_a_name_no_role_says_is_thrown_away_however_many_replies_give_it(monkeypatch):
    """Every reply says Terraform is shown by Pulumi, and this resume never says Pulumi."""
    _model_replies(monkeypatch, _reply(another_way=[("Terraform", "Pulumi")]))

    keywords = (await rubric.ats_score(RESUME_WITH_OPENTOFU, POSTING))["keywords"]

    assert keywords["shown_another_way"] == []
    assert keywords["listed"] == ["Terraform"]


@pytest.mark.asyncio
async def test_a_name_only_in_the_skills_list_is_not_a_role_showing_it(monkeypatch):
    _model_replies(monkeypatch, _reply(another_way=[("Datadog", "Linux")]))        # Linux is in the skills list only

    assert (await rubric.ats_score(RESUME, POSTING))["keywords"]["shown_another_way"] == []


def test_only_the_names_a_role_writes_are_kept_from_what_the_model_quotes():
    work = rubric._as_plain_words("- Built clusters on Amazon EKS and Azure AKS\n- Wrote Terraform modules")

    assert rubric.names_written_in_the_roles("Terraform", work) == "Terraform"
    assert rubric.names_written_in_the_roles("terraform modules", work) == "terraform modules"
    assert rubric.names_written_in_the_roles("EKS, AKS and GKE", work) == "EKS, AKS"
    assert rubric.names_written_in_the_roles("Google GKE", work) == ""
    assert rubric.names_written_in_the_roles("", work) == ""
    assert rubric.names_written_in_the_roles("KS", work) == ""           # part of a word is not the word


@pytest.mark.asyncio
async def test_input_served_from_the_providers_cache_is_counted_as_input(monkeypatch):
    reply = {**_reply(), "usage": {"input_tokens": 2, "cache_read_tokens": 9774, "cache_write_tokens": 297, "output_tokens": 4356}}
    _model_replies(monkeypatch, reply)

    result = await rubric.ats_score(RESUME, POSTING)

    assert (result["input_tokens"], result["output_tokens"]) == (3 * 10073, 3 * 4356)
