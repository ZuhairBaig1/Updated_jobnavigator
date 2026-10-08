"""A posting's must-have lines and preferred lines: found from its headings, then checked line by line."""
import pytest

from backend.analyzer import decision_scorer, line_match

POSTING = """About us
We build payment software for banks.

Requirements
- Python
- Experience managing multiple active versions of a product in production
- Kafka would be a plus

Location: Pune
"""


def test_the_posting_is_cut_into_its_own_lines_with_nothing_reworded():
    assert line_match.posting_lines(POSTING) == [
        "About us", "We build payment software for banks.", "Requirements", "Python",
        "Experience managing multiple active versions of a product in production", "Kafka would be a plus",
        "Location: Pune"]


def test_a_long_paragraph_is_cut_into_sentences():
    paragraph = ("Strong expertise in GenAI frameworks and the wider set of AI platforms used by enterprise clients "
                 "today. Design and deliver agentic AI architectures for enterprise clients across several industries.")

    assert len(line_match.posting_lines(paragraph)) == 2


def test_the_state_is_the_plain_text_under_a_short_label():
    assert line_match._posting_state("We need Python.") == "Job posting: We need Python."


# ── Headings, and the lines under them ─────────────────────────────────────────

HEADINGS_POSTING = """Role: DevOps Engineer
Mandatory Skills
- Kubernetes
- Docker
Preferred Qualifications
- AWS DevOps Certification
We are an equal opportunity employer.
"""
THE_HEADINGS = ["Mandatory Skills", "Preferred Qualifications"]
LINES_THAT_DO_NOT_BELONG = ["We are an equal opportunity employer."]


def _jev_reading_headings(monkeypatch, headings=THE_HEADINGS, do_not_belong=LINES_THAT_DO_NOT_BELONG):
    calls = []

    def quoted_line(question):
        parts = question["instructions"].split('"')
        return parts[1] if len(parts) > 1 else ""

    async def fake_ask_model(scorer, api_key, state, questions):
        calls.append({"state": state, "questions": questions})
        answers = {}
        for key, question in questions.items():
            line = quoted_line(question)
            if "a heading:" in question["instructions"]:
                answers[key] = {"choice": "heading" if line in headings else "not_a_heading"}
            else:
                answers[key] = {"choice": "does_not_belong" if line in do_not_belong else "belongs"}
        return {"answers": answers}
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "key")
    monkeypatch.setattr(decision_scorer, "_ask_model", fake_ask_model)
    return calls


@pytest.mark.asyncio
async def test_the_headings_are_the_lines_jev_calls_headings(monkeypatch):
    _jev_reading_headings(monkeypatch)

    result = await line_match.lines_under_headings("jev", HEADINGS_POSTING)

    assert result["headings"] == THE_HEADINGS


@pytest.mark.asyncio
async def test_a_line_gets_the_heading_above_it_by_position(monkeypatch):
    _jev_reading_headings(monkeypatch)

    result = await line_match.lines_under_headings("jev", HEADINGS_POSTING)

    assert [(row["line"], row["heading"]) for row in result["lines"][:4]] == [
        ("Role: DevOps Engineer", None),
        ("Kubernetes", "Mandatory Skills"),
        ("Docker", "Mandatory Skills"),
        ("AWS DevOps Certification", "Preferred Qualifications"),
    ]
    assert result["heading_numbers"] == [1, 4]
    assert result["lines"][1]["number"] == 2 and result["lines"][1]["heading_number"] == 1


@pytest.mark.asyncio
async def test_a_line_that_only_sits_below_a_heading_loses_it(monkeypatch):
    _jev_reading_headings(monkeypatch)

    result = await line_match.lines_under_headings("jev", HEADINGS_POSTING)

    last_row = result["lines"][-1]
    assert (last_row["line"], last_row["heading"]) == ("We are an equal opportunity employer.", None)


@pytest.mark.asyncio
async def test_jev_is_asked_about_a_line_and_the_heading_above_it_together(monkeypatch):
    calls = _jev_reading_headings(monkeypatch)

    await line_match.lines_under_headings("jev", HEADINGS_POSTING)

    # One call for the headings, one for belonging, and one more for the line that was turned away.
    assert [call["state"] for call in calls] == ["Job posting: " + HEADINGS_POSTING] * 3
    questions_about_belonging = [question["instructions"] for question in calls[1]["questions"].values()]
    assert questions_about_belonging[0] == (
        'In the job posting, the line "Kubernetes" sits below the heading "Mandatory Skills". '
        "Is the line part of that section?")
    # The line before the first heading has no heading to belong to, so nothing is asked about it.
    assert not any("Role: DevOps Engineer" in question for question in questions_about_belonging)


@pytest.mark.asyncio
async def test_a_posting_with_no_headings_leaves_every_line_without_one(monkeypatch):
    calls = _jev_reading_headings(monkeypatch, headings=[])

    result = await line_match.lines_under_headings("jev", HEADINGS_POSTING)

    assert result["headings"] == []
    assert all(row["heading"] is None for row in result["lines"])
    assert len(calls) == 1


# ── Must-haves and preferred skills ────────────────────────────────────────────

SKILLS_POSTING = """Mandatory Skills
- Kubernetes
- Cloud infrastructure (Azure preferred)
- AWS certification is an advantage
Preferred Qualifications
- Terraform
Key Responsibilities
- Build pipelines
"""
SECTIONS = {"Mandatory Skills": "requirements", "Preferred Qualifications": "preferred", "Key Responsibilities": "other"}


def _jev_sorting_skills(monkeypatch, headings=tuple(SECTIONS), sections=SECTIONS, places=None, coverage=None,
                        kinds=None, duty_headings=("Key Responsibilities",), occupation=5,
                        importance=None):
    """A pretend Jev. `places` maps a line to (the place it picks, how sure it is); other lines stay put."""
    def quoted_line(question):
        parts = question["instructions"].split('"')
        return parts[1] if len(parts) > 1 else ""

    def place_of(line, asked):
        place_from_the_heading = "must_have" if "part of the must-haves" in asked else "preferred"
        place, how_sure = (places or {}).get(line, (place_from_the_heading, 1.0))
        likelihoods = {"must_have": 0.0, "preferred": 0.0, "other": 0.0}
        likelihoods[place] = how_sure
        if place != place_from_the_heading:
            likelihoods[place_from_the_heading] = 1.0 - how_sure
        return {"choice": place, "probabilities": likelihoods}

    async def fake_ask_model(scorer, api_key, state, questions):
        answers = {}
        for key, question in questions.items():
            line, asked = quoted_line(question), question["instructions"]
            if "same occupation" in asked:
                answers[key] = {"score": occupation}
            elif "How important is it to this job" in asked:
                answers[key] = {"score": (importance or {}).get(line, 0)}
            elif question["type"] == "score":
                answers[key] = {"score": (coverage or {}).get(line, 0)}
            elif "a heading:" in asked:
                answers[key] = {"choice": "heading" if line in headings else "not_a_heading"}
            elif "sits below the heading" in asked:
                answers[key] = {"choice": "belongs"}
            elif "Are those lines the duties" in asked:
                answers[key] = {"choice": "duties" if line in duty_headings else "not_duties"}
            elif "What kind of requirement" in asked:
                answers[key] = {"choice": (kinds or {}).get(line, "skill")}
            elif "has these lines under it" in asked:
                answers[key] = {"choice": sections[line]}
            else:
                answers[key] = place_of(line, asked)
        return {"answers": answers}
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "key")
    monkeypatch.setattr(decision_scorer, "_ask_model", fake_ask_model)


@pytest.mark.asyncio
async def test_lines_under_requirement_and_preferred_headings_are_used(monkeypatch):
    _jev_sorting_skills(monkeypatch)

    result = await line_match.must_haves_and_preferred("jev", SKILLS_POSTING)

    assert "Kubernetes" in result["must_have"]
    assert "Terraform" in result["preferred"]
    assert "Build pipelines" not in result["must_have"] + result["preferred"]
    assert result["found_from"] == "headings"


@pytest.mark.asyncio
async def test_a_must_have_line_moves_to_preferred_when_jev_is_sure(monkeypatch):
    _jev_sorting_skills(monkeypatch, places={"AWS certification is an advantage": ("preferred", 0.97)})

    result = await line_match.must_haves_and_preferred("jev", SKILLS_POSTING)

    assert "AWS certification is an advantage" in result["preferred"]
    assert "AWS certification is an advantage" not in result["must_have"]
    assert "Cloud infrastructure (Azure preferred)" in result["must_have"]


@pytest.mark.asyncio
async def test_a_line_stays_where_its_heading_put_it_when_jev_is_not_sure(monkeypatch):
    _jev_sorting_skills(monkeypatch, places={"AWS certification is an advantage": ("preferred", 0.70),
                                            "Terraform": ("must_have", 0.80),
                                            "Kubernetes": ("other", 0.55)})

    result = await line_match.must_haves_and_preferred("jev", SKILLS_POSTING)

    assert "AWS certification is an advantage" in result["must_have"]
    assert "Kubernetes" in result["must_have"]
    assert result["preferred"] == ["Terraform"]


@pytest.mark.asyncio
async def test_a_line_jev_is_sure_is_not_about_the_candidate_is_left_out(monkeypatch):
    _jev_sorting_skills(monkeypatch, places={"Kubernetes": ("other", 0.98)})

    result = await line_match.must_haves_and_preferred("jev", SKILLS_POSTING)

    assert "Kubernetes" not in result["must_have"] + result["preferred"]


@pytest.mark.asyncio
async def test_the_section_question_lists_the_lines_under_the_heading(monkeypatch):
    asked = []
    _jev_sorting_skills(monkeypatch)
    ask = decision_scorer._ask_model

    async def remembering(scorer, api_key, state, questions):
        asked.extend(question["instructions"] for question in questions.values())
        return await ask(scorer, api_key, state, questions)
    monkeypatch.setattr(decision_scorer, "_ask_model", remembering)

    await line_match.must_haves_and_preferred("jev", SKILLS_POSTING)

    section_questions = [question for question in asked if "has these lines under it" in question]
    assert any('"Preferred Qualifications" has these lines under it: Terraform.' in question for question in section_questions)
    assert any('the line "Terraform" is under the heading "Preferred Qualifications", which is part of the preferred'
               in question for question in asked)


@pytest.mark.asyncio
async def test_preferred_inside_brackets_does_not_make_the_whole_line_optional(monkeypatch):
    _jev_sorting_skills(monkeypatch)

    result = await line_match.must_haves_and_preferred("jev", SKILLS_POSTING)

    assert "Cloud infrastructure (Azure preferred)" in result["must_have"]


NO_HEADINGS_POSTING = """We need a platform engineer for our payments team.
Strong experience with Kubernetes and Terraform.
Kafka would be a plus.
We offer free lunch.
"""


@pytest.mark.asyncio
async def test_a_posting_with_no_headings_gives_empty_lists_and_says_nothing_was_found(monkeypatch):
    _jev_sorting_skills(monkeypatch, headings=())

    result = await line_match.must_haves_and_preferred("jev", NO_HEADINGS_POSTING)

    assert result == {"must_have": [], "preferred": [], "found_from": "nothing"}


@pytest.mark.asyncio
async def test_a_posting_whose_sections_are_all_other_also_says_nothing_was_found(monkeypatch):
    posting = "Key Responsibilities\n- Build pipelines\nStrong experience with Kubernetes and Terraform.\n"
    _jev_sorting_skills(monkeypatch, headings=("Key Responsibilities",))

    result = await line_match.must_haves_and_preferred("jev", posting)

    assert result == {"must_have": [], "preferred": [], "found_from": "nothing"}


@pytest.mark.asyncio
async def test_nothing_is_returned_when_jev_cannot_be_reached(monkeypatch):
    async def failing_ask_model(scorer, api_key, state, questions):
        raise RuntimeError("server error")
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "key")
    monkeypatch.setattr(decision_scorer, "_ask_model", failing_ask_model)

    assert await line_match.must_haves_and_preferred("jev", NO_HEADINGS_POSTING) is None


LEAD_IN_POSTING = """Acme builds payment software.
To set you up for success, you will bring the following skills:
    Kubernetes
    Terraform
Whilst these are nice to have, we can help you develop them:
    Kafka
"""
LEAD_IN_SECTIONS = {"To set you up for success, you will bring the following skills:": "requirements",
                    "Whilst these are nice to have, we can help you develop them:": "preferred"}


@pytest.mark.asyncio
async def test_a_sentence_that_introduces_a_list_is_a_heading_even_when_jev_misses_it(monkeypatch):
    _jev_sorting_skills(monkeypatch, headings=(), sections=LEAD_IN_SECTIONS)

    result = await line_match.must_haves_and_preferred("jev", LEAD_IN_POSTING)

    assert result == {"must_have": ["Kubernetes", "Terraform"], "preferred": ["Kafka"], "found_from": "headings"}


def test_a_colon_line_is_a_lead_in_only_when_a_list_follows_it():
    posting = "Location: Pune\nSkills needed:\n- Python\nNote:\nApply soon."

    assert line_match._lines_that_introduce_a_list(posting) == ["Skills needed:"]


PREFERRED_HEADINGS_POSTING = """Requirements
- Kubernetes
Nice to have skills
- Kafka
Required and preferred skills
- Python
"""
PREFERRED_HEADINGS = ("Requirements", "Nice to have skills", "Required and preferred skills")
# Jev answers "requirements" for every heading here, so only the wording can make anything preferred.
ALL_REQUIREMENTS = {heading: "requirements" for heading in PREFERRED_HEADINGS}


@pytest.mark.asyncio
async def test_what_the_posting_states_as_preferred_is_preferred_whatever_jev_answers(monkeypatch):
    _jev_sorting_skills(monkeypatch, headings=PREFERRED_HEADINGS, sections=ALL_REQUIREMENTS)

    result = await line_match.must_haves_and_preferred("jev", PREFERRED_HEADINGS_POSTING)

    assert result["preferred"] == ["Kafka"]
    assert result["must_have"] == ["Kubernetes", "Python"]


def test_a_heading_about_the_preferred_candidate_is_left_to_jev():
    under_headings = {"heading_numbers": [0, 5], "headings": ["Preferred candidate profile", "Preferred skills"]}
    section_of = {0: "requirements", 5: "requirements"}

    line_match._headings_that_say_preferred_are_preferred(section_of, under_headings)

    assert section_of == {0: "requirements", 5: "preferred"}


def test_jev_is_shown_the_whole_of_a_long_posting():
    long_posting = "Requirements\n" + "Python\n" * 3000 + "Preferred Qualifications\nKafka"

    assert line_match._posting_state(long_posting).endswith("Preferred Qualifications\nKafka")


# ── Cutting first, cleaning second ─────────────────────────────────────────────

def test_a_bold_heading_glued_to_the_end_of_a_sentence_becomes_its_own_line():
    posting = "This is the place to be.**BASIC QUALIFICATIONS**\n--------\n\n* 3\\+ years of experience"

    assert line_match.posting_lines(posting) == ["This is the place to be.", "BASIC QUALIFICATIONS", "3+ years of experience"]
    assert line_match.headings_the_posting_declares(posting) == ["BASIC QUALIFICATIONS"]


def test_a_bold_label_is_cut_from_the_content_that_follows_it():
    assert line_match.posting_lines("**Must have skills :** Generative AI") == ["Must have skills :", "Generative AI"]
    assert line_match.posting_lines("**PREFERRED EXPERIENCE:*** Strong C++ background") == [
        "PREFERRED EXPERIENCE:", "Strong C++ background"]
    assert line_match.posting_lines("**Required Skills:****Core Domain:*** Bachelor's degree") == [
        "Required Skills: Core Domain:", "Bachelor's degree"]


def test_a_heading_written_as_two_bold_runs_is_one_heading():
    assert line_match.posting_lines("**Preferred****Qualifications:**") == ["Preferred Qualifications:"]


def test_a_cut_label_that_is_not_a_heading_rejoins_its_content():
    lines = ["Requirements", "Deep experience:", "8+ years building systems", "Must have skills :", "Generative AI"]
    content_after = {"Deep experience:": "8+ years building systems", "Must have skills :": "Generative AI"}

    rejoined, heading_numbers = line_match._labels_that_are_not_headings_put_back(
        lines, heading_numbers={0, 3}, content_after=content_after)

    assert rejoined == ["Requirements", "Deep experience: 8+ years building systems", "Must have skills :", "Generative AI"]
    # "Must have skills :" was line 3 and is line 2 now that two lines above it were joined.
    assert heading_numbers == {0, 2}


def test_bold_words_inside_a_sentence_or_a_list_item_stay_in_their_line():
    assert line_match.posting_lines("Minimum **5** year(s) of experience is required") == [
        "Minimum 5 year(s) of experience is required"]
    assert line_match.posting_lines("* **Communication:** Ability to explain ideas") == [
        "Communication: Ability to explain ideas"]
    assert line_match.posting_lines("Experience in **Python**") == ["Experience in Python"]


def test_hash_and_underlined_lines_are_declared_headings_and_a_rule_after_a_blank_line_is_not():
    posting = "# Senior Engineer\nSome text about the job.\n\n-----\nRequirements\n=====\nPython"

    assert line_match.headings_the_posting_declares(posting) == ["Senior Engineer", "Requirements"]


def test_a_posting_saved_as_a_quoted_string_is_read_with_its_line_breaks_and_marks():
    saved = '"## Job description\\n\\n**Required Skills**\\n\\n- Docker&#x20;and Kubernetes\\n- AWS &amp; Azure"'

    assert line_match.posting_lines(saved) == ["Job description", "Required Skills", "Docker and Kubernetes", "AWS & Azure"]
    assert line_match.headings_the_posting_declares(saved) == ["Job description"]


def test_a_posting_with_real_line_breaks_keeps_a_backslash_n_it_mentions():
    posting = "Requirements\nKnows that \\n means a line break"

    assert line_match.posting_lines(posting) == ["Requirements", "Knows that \\n means a line break"]


def test_links_are_removed_before_anything_reads_the_posting():
    posting = ("Key Skills\n[Docker](https://www.naukri.com/docker-jobs)Ci/Cd[AWS](https://www.naukri.com/aws-jobs)Kubernetes\n"
               "Role: [DevOps Engineer](https://www.naukri.com/devops-engineer-jobs)\nApply at https://example.com/jobs/42 today")

    assert line_match.posting_lines(posting) == [
        "Key Skills", "Docker Ci/Cd AWS Kubernetes", "Role: DevOps Engineer", "Apply at today"]
    assert "http" not in line_match._posting_state(posting)


def test_characters_saved_as_codes_in_a_quoted_posting_are_read_as_characters():
    saved = '"Required Skills\\n- Docker\\u00a0\\n- Caf\\u00e9 systems"'

    assert line_match.posting_lines(saved) == ["Required Skills", "Docker", "Café systems"]


def test_the_job_boards_key_skills_at_the_end_are_left_out_and_the_employers_own_are_kept():
    posting = ("Key Skills\n- Kubernetes\n- Docker\nResponsibilities\n- Run clusters\n"
               "Role Category: DevOps\nEducation\nUG: Any Graduate\nKey Skills\n"
               "Skills highlighted with ‘‘ are preferred keyskills\nDocker Ci/Cd AWS")

    assert line_match.posting_lines(posting) == [
        "Key Skills", "Kubernetes", "Docker", "Responsibilities", "Run clusters", "Role Category: DevOps",
        "Education", "UG: Any Graduate"]


def test_a_key_skills_section_with_no_job_board_box_above_it_is_kept():
    posting = "Senior Engineer\nKey Skills\n- Python\n- AWS"

    assert line_match.posting_lines(posting) == ["Senior Engineer", "Key Skills", "Python", "AWS"]


def test_the_job_boards_preferred_candidate_profile_heading_is_made_neutral():
    posting = "**Preferred candidate profile**\n- Strong AWS experience\nPreferred Qualifications\n- Kafka"

    assert line_match.posting_lines(posting) == [
        "Candidate profile", "Strong AWS experience", "Preferred Qualifications", "Kafka"]
    assert "Preferred candidate profile" not in line_match._posting_state(posting)


def test_a_sentence_that_mentions_the_preferred_candidate_profile_is_left_alone():
    posting = "Our preferred candidate profile is a hands-on engineer."

    assert line_match.posting_lines(posting) == [posting]


LABELLED_POSTING = """Mandatory Skills
- Primary Skill : Devops, Azure
- Secondary: Terraform, Ansible
- Preferred: Kafka
- Good to have skills - Helm
- Strong experience with cloud platforms (AWS preferred)
- Preferred shift is the night shift
"""


@pytest.mark.asyncio
async def test_a_line_labelled_secondary_or_preferred_is_preferred_whatever_jev_answers(monkeypatch):
    _jev_sorting_skills(monkeypatch, headings=("Mandatory Skills",), sections={"Mandatory Skills": "requirements"})

    result = await line_match.must_haves_and_preferred("jev", LABELLED_POSTING)

    assert result["preferred"] == ["Secondary: Terraform, Ansible", "Preferred: Kafka", "Good to have skills - Helm"]
    assert result["must_have"] == ["Primary Skill : Devops, Azure",
                                   "Strong experience with cloud platforms (AWS preferred)",
                                   "Preferred shift is the night shift"]


# ── Lines are told apart by where they are, not by what they say ───────────────

REPEATED_NAME_POSTING = """Responsibilities
- Automate deployments for applications using:
    - S3
    - DynamoDB
Required Skills
- Working knowledge of:
    - S3
    - DynamoDB
- Experience with Git
"""


def test_a_line_that_appears_in_two_sections_is_kept_in_both():
    assert line_match.posting_lines(REPEATED_NAME_POSTING).count("S3") == 2


def test_a_long_run_of_lines_that_repeats_is_dropped_the_second_time():
    page = [f"Line {number} of the posting" for number in range(12)]

    assert line_match.posting_lines("\n".join(page + page)) == page


def _jev_reading_a_sub_list(monkeypatch):
    """Duties are "other" and Required Skills is must-have. "Experience with Git" does not belong to the sub-list."""
    # Jev judges both sub-lists on their own and gets them backwards; each must take its section's judgement.
    sections = {"Responsibilities": "other", "Automate deployments for applications using:": "requirements",
                "Required Skills": "requirements", "Working knowledge of:": "other"}

    async def fake_ask_model(scorer, api_key, state, questions):
        answers = {}
        for key, question in questions.items():
            asked = question["instructions"]
            line = asked.split('"')[1]
            if "a heading:" in asked:
                answers[key] = {"choice": "heading" if line in sections else "not_a_heading"}
            elif "sits below the heading" in asked:
                turned_away = line == "Experience with Git" and '"Working knowledge of:"' in asked
                answers[key] = {"choice": "does_not_belong" if turned_away else "belongs"}
            elif "has these lines under it" in asked:
                answers[key] = {"choice": sections[line]}
            else:
                answers[key] = {"choice": "must_have", "probabilities": {"must_have": 1.0, "preferred": 0.0, "other": 0.0}}
        return {"answers": answers}
    monkeypatch.setattr(decision_scorer, "_find_api_key", lambda scorer: "key")
    monkeypatch.setattr(decision_scorer, "_ask_model", fake_ask_model)


@pytest.mark.asyncio
async def test_a_line_after_a_sub_list_goes_back_to_the_heading_before_it(monkeypatch):
    _jev_reading_a_sub_list(monkeypatch)

    result = await line_match.lines_under_headings("jev", REPEATED_NAME_POSTING)

    git_row = next(row for row in result["lines"] if row["line"] == "Experience with Git")
    assert git_row["heading"] == "Required Skills"


@pytest.mark.asyncio
async def test_a_name_repeated_from_the_duties_is_still_a_must_have_under_required_skills(monkeypatch):
    _jev_reading_a_sub_list(monkeypatch)

    result = await line_match.must_haves_and_preferred("jev", REPEATED_NAME_POSTING)

    assert result["must_have"] == ["S3", "DynamoDB", "Experience with Git"]
    assert result["preferred"] == []


def test_a_list_item_that_introduces_a_deeper_list_is_a_sub_heading():
    assert line_match._texts_of_sub_headings(REPEATED_NAME_POSTING) == {
        "Automate deployments for applications using:", "Working knowledge of:"}


def test_a_heading_followed_by_items_at_the_same_depth_is_not_a_sub_heading():
    posting = "- Qualifications:\n- 5 years of experience\n- Python"

    assert line_match._texts_of_sub_headings(posting) == set()


def test_the_first_heading_keeps_its_own_judgement_even_when_it_is_a_sub_heading():
    under_headings = {"heading_numbers": [0, 4], "sub_heading_numbers": [0, 4]}
    section_of = {0: "requirements", 4: "other"}

    line_match._sub_headings_take_the_section_they_sit_in(section_of, under_headings)

    assert section_of == {0: "requirements", 4: "requirements"}


@pytest.mark.asyncio
async def test_a_line_is_left_out_at_a_lower_bar_than_it_takes_to_move_it(monkeypatch):
    _jev_sorting_skills(monkeypatch, places={"Kubernetes": ("other", 0.65),
                                            "AWS certification is an advantage": ("preferred", 0.65)})

    result = await line_match.must_haves_and_preferred("jev", SKILLS_POSTING)

    assert "Kubernetes" not in result["must_have"] + result["preferred"]
    assert "AWS certification is an advantage" in result["must_have"]


EDUCATION_POSTING = """Education
UG: Any Graduate
PG: Any Postgraduate
"""


@pytest.mark.asyncio
async def test_the_job_boards_education_lines_are_kept_even_when_jev_would_leave_them_out(monkeypatch):
    _jev_sorting_skills(monkeypatch, headings=("Education",), sections={"Education": "requirements"},
                        places={"UG: Any Graduate": ("other", 0.70), "PG: Any Postgraduate": ("other", 0.95)})

    result = await line_match.must_haves_and_preferred("jev", EDUCATION_POSTING)

    assert result["must_have"] == ["UG: Any Graduate", "PG: Any Postgraduate"]


@pytest.mark.asyncio
async def test_an_education_line_that_says_no_degree_is_required_can_be_left_out(monkeypatch):
    _jev_sorting_skills(monkeypatch, headings=("Education",), sections={"Education": "requirements"},
                        places={"UG: Graduation Not Required": ("other", 0.95)})

    result = await line_match.must_haves_and_preferred("jev", "Education\nUG: Graduation Not Required\n")

    assert result == {"must_have": [], "preferred": [], "found_from": "nothing"}


# ── The years of experience, wherever the posting states them ──────────────────

YEARS_IN_THE_HEADER_POSTING = """DevOps Senior Engineer | Hyderabad
Role: DevOps Senior Engineer
Location: Hyderabad (On-site)
Experience: 10-15 Years
Mandatory Skills
- Kubernetes
"""


@pytest.mark.asyncio
async def test_the_years_of_experience_in_the_header_are_added_to_the_must_haves(monkeypatch):
    headings = ("DevOps Senior Engineer | Hyderabad", "Mandatory Skills")
    sections = {"DevOps Senior Engineer | Hyderabad": "other", "Mandatory Skills": "requirements"}
    _jev_sorting_skills(monkeypatch, headings=headings, sections=sections)

    result = await line_match.must_haves_and_preferred("jev", YEARS_IN_THE_HEADER_POSTING)

    assert result["must_have"] == ["Experience: 10-15 Years", "Kubernetes"]


@pytest.mark.asyncio
async def test_a_years_line_that_is_already_a_must_have_is_not_added_twice(monkeypatch):
    posting = "Candidate profile\n- Exp: 6 to 12\n- Kubernetes\n"
    _jev_sorting_skills(monkeypatch, headings=("Candidate profile",), sections={"Candidate profile": "requirements"})

    result = await line_match.must_haves_and_preferred("jev", posting)

    assert result["must_have"] == ["Exp: 6 to 12", "Kubernetes"]


def test_which_lines_state_the_years_of_experience():
    for line in ("Experience: 10-15 Years", "Exp: 6 to 12", "Experience Level: 5+ Years", "Experience Range: 2 - 3 years",
                 "Experience: Minimum 5 years", "Total Experience - 8 years"):
        assert line_match._states_the_years_of_experience(line), line
    for line in ("Experience in Production Support / SRE / Cloud Operations.", "Experience Profile:", "Location: Pune",
                 "Experience with Docker: 2 containers per pod is typical in our setup and you will be asked to run "
                 "many more of them in production"):
        assert not line_match._states_the_years_of_experience(line), line


def test_stray_backslashes_and_broken_character_codes_are_cleaned_off_a_line():
    saved = ('"## Job description\\n\\n***Role:** DevOps Senior Engineer*\\\\&#xA;*\\\\x20;**Experience:** 10-15 Years*'
             '\\\\&#xA;*\\\\x20;**Type:** Full-Time*"')

    assert line_match.posting_lines(saved) == [
        "Job description", "Role: DevOps Senior Engineer", "Experience: 10-15 Years", "Type: Full-Time"]


# ── How much of each line the résumé shows ─────────────────────────────────────

def _tools_by_plain_lookup(monkeypatch, tools=("Kubernetes", "Azure", "AWS", "Terraform", "Docker")):
    """Stands in for the skills list: a line names a tool when the tool's name is written in it."""
    monkeypatch.setattr(line_match, "_tools_named_in", lambda line: {tool for tool in tools if tool in line})


@pytest.mark.asyncio
async def test_each_must_have_and_preferred_line_gets_the_share_the_resume_covers(monkeypatch):
    # Jev's levels run from 0 (none) to 4 (everything): 4 is all of the line, 2 is about half.
    _jev_sorting_skills(monkeypatch, coverage={"Kubernetes": 4, "Cloud infrastructure (Azure preferred)": 2, "Terraform": 1})
    _tools_by_plain_lookup(monkeypatch)

    result = await line_match.resume_against_the_posting("jev", "Ran Kubernetes clusters.", SKILLS_POSTING)

    assert result["must_have"] == [{"line": "Kubernetes", "covered": 1.0, "weight": 1.0},
                                   {"line": "Cloud infrastructure (Azure preferred)", "covered": 0.5, "weight": 1.0},
                                   {"line": "AWS certification is an advantage", "covered": 0.0, "weight": 1.0}]
    assert result["preferred"] == [{"line": "Terraform", "covered": 0.25, "weight": 1.0}]
    assert result["must_have_covered"] == 0.5
    assert result["preferred_covered"] == 0.25


def test_a_tool_named_in_several_lines_counts_once_and_a_line_with_no_tool_counts_as_one(monkeypatch):
    _tools_by_plain_lookup(monkeypatch)
    lines = ["Kubernetes", "Experience with Kubernetes and Docker", "5+ years of experience", "AWS, Azure, Terraform"]

    weights = line_match._weights_so_each_tool_counts_once(lines)

    # Kubernetes is shared by two lines, so each gets half of it. Docker adds one to the second line.
    assert weights == [0.5, 1.5, 1.0, 3.0]


def test_a_tool_that_is_a_must_have_is_not_counted_again_under_preferred(monkeypatch):
    _tools_by_plain_lookup(monkeypatch)

    weights = line_match._weights_so_each_tool_counts_once(["Kubernetes", "Terraform and Docker", "Good communication"],
                                                           tools_counted_already={"Kubernetes", "Docker"})

    assert weights == [0.0, 1.0, 1.0]


def test_the_average_counts_each_line_by_its_weight():
    rows = [{"line": "a", "covered": 1.0, "weight": 3.0}, {"line": "b", "covered": 0.0, "weight": 1.0}]

    assert line_match._average_coverage(rows) == 0.75
    assert line_match._average_coverage([{"line": "c", "covered": 1.0, "weight": 0.0}]) is None


@pytest.mark.asyncio
async def test_jev_reads_the_resume_and_the_posting_when_it_scores_a_line(monkeypatch):
    states = []
    _jev_sorting_skills(monkeypatch)
    ask = decision_scorer._ask_model

    async def remembering(scorer, api_key, state, questions):
        if any(question["type"] == "score" for question in questions.values()):
            states.append(state)
        return await ask(scorer, api_key, state, questions)
    monkeypatch.setattr(decision_scorer, "_ask_model", remembering)

    await line_match.resume_against_the_posting("jev", "Ran Kubernetes clusters.", SKILLS_POSTING)

    assert states == ["Resume: Ran Kubernetes clusters.\n\nJob posting: " + SKILLS_POSTING]


@pytest.mark.asyncio
async def test_a_posting_with_nothing_found_gives_no_averages(monkeypatch):
    _jev_sorting_skills(monkeypatch, headings=())

    result = await line_match.resume_against_the_posting("jev", "Ran Kubernetes clusters.", NO_HEADINGS_POSTING)

    assert result == {"must_have": [], "preferred": [], "must_have_covered": None, "preferred_covered": None,
                      "found_from": "nothing"}


# ── The lists, with the duties ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_lines_of_a_duties_section_are_kept_as_duties(monkeypatch):
    _jev_sorting_skills(monkeypatch)

    lists = await line_match.lines_of_the_posting_by_what_they_are("jev", SKILLS_POSTING)

    assert lists["duties"] == ["Build pipelines"]
    assert "Build pipelines" not in lists["must_have"] + lists["preferred"]


@pytest.mark.asyncio
async def test_a_section_that_is_neither_skills_nor_duties_is_left_out(monkeypatch):
    _jev_sorting_skills(monkeypatch, duty_headings=())

    lists = await line_match.lines_of_the_posting_by_what_they_are("jev", SKILLS_POSTING)

    assert lists["duties"] == []


# ── The match score ────────────────────────────────────────────────────────────

def _role_answers_are(monkeypatch, role_fit, experience=1.0):
    async def fake_score_resume(scorer, resume_text, job_text, requirements=None):
        return {"answers": {"core_match": role_fit, "experience_match": experience}}
    monkeypatch.setattr(decision_scorer, "score_resume", fake_score_resume)


def test_the_score_is_role_fit_times_a_fixed_part_plus_the_total_of_the_parts():
    assert line_match._score_out_of_100(role_fit=1.0, total_of_parts=1.0) == 100
    assert line_match._score_out_of_100(role_fit=1.0, total_of_parts=0.0) == 40
    assert line_match._score_out_of_100(role_fit=0.4, total_of_parts=0.75) == 34
    assert line_match._score_out_of_100(role_fit=0.9, total_of_parts=None) is None


def test_the_parts_are_added_by_their_weights():
    parts = {"skills": 0.8, "responsibilities": 0.6, "experience": 1.0, "qualifications": 0.5}

    # 0.45 x 0.8 + 0.25 x 0.6 + 0.15 x 1.0 + 0.15 x 0.5
    assert line_match._parts_added_by_weight(parts) == 0.73


def test_a_part_the_posting_says_nothing_about_gives_its_weight_to_the_others():
    parts = {"skills": 0.8, "responsibilities": None, "experience": 1.0, "qualifications": None}

    # Only skills (0.45) and experience (0.15) are scored: (0.45 x 0.8 + 0.15 x 1.0) / 0.60
    assert line_match._parts_added_by_weight(parts) == 0.85
    assert line_match._parts_added_by_weight({part: None for part in parts}) is None


def test_must_haves_count_twice_as_much_as_preferred_in_a_part():
    assert round(line_match._coverage_of_both_lists(0.9, 0.3), 2) == 0.7
    assert line_match._coverage_of_both_lists(0.9, None) == 0.9
    assert line_match._coverage_of_both_lists(None, None) is None


@pytest.mark.asyncio
async def test_the_match_score_has_a_part_for_each_kind_of_line(monkeypatch):
    coverage = {"Kubernetes": 4, "Cloud infrastructure (Azure preferred)": 2, "AWS certification is an advantage": 0,
                "Terraform": 1, "Build pipelines": 3}
    kinds = {"AWS certification is an advantage": "qualification"}
    _jev_sorting_skills(monkeypatch, coverage=coverage, kinds=kinds)
    _tools_by_plain_lookup(monkeypatch)
    _role_answers_are(monkeypatch, role_fit=0.5, experience=0.8)

    result = await line_match.match_score("jev", "Ran Kubernetes clusters.", SKILLS_POSTING)

    # Skills: must-haves (1.0 + 0.5) / 2 = 0.75, preferred 0.25 -> (2 x 0.75 + 0.25) / 3 = 0.58
    assert result["parts"] == {"skills": 0.58, "responsibilities": 0.75, "experience": 0.8, "qualifications": 0.0}
    # 0.45 x 0.583 + 0.25 x 0.75 + 0.15 x 0.8 + 0.15 x 0 = 0.57
    assert result["total_of_parts"] == 0.57
    assert result["score"] == round(100 * 0.5 * (0.4 + 0.6 * 0.57))
    assert result["duties"] == [{"line": "Build pipelines", "covered": 0.75}]
    assert result["must_have"][2]["kind"] == "qualification"


@pytest.mark.asyncio
async def test_a_resume_of_a_different_kind_of_work_scores_low_even_with_every_line_covered(monkeypatch):
    every_line = {"Kubernetes": 4, "Cloud infrastructure (Azure preferred)": 4, "AWS certification is an advantage": 4,
                  "Terraform": 4, "Build pipelines": 4}
    _jev_sorting_skills(monkeypatch, coverage=every_line)
    _tools_by_plain_lookup(monkeypatch)
    _role_answers_are(monkeypatch, role_fit=0.35)

    result = await line_match.match_score("jev", "Lists every tool.", SKILLS_POSTING)

    assert result["total_of_parts"] == 1.0
    assert result["score"] == 35


@pytest.mark.asyncio
async def test_a_posting_with_no_lines_to_score_gives_no_score(monkeypatch):
    _jev_sorting_skills(monkeypatch, headings=())
    _role_answers_are(monkeypatch, role_fit=0.9)

    result = await line_match.match_score("jev", "Ran Kubernetes clusters.", NO_HEADINGS_POSTING)

    assert result["score"] is None
    assert result["parts"] == {"skills": None, "responsibilities": None, "experience": None, "qualifications": None}


# ── The ATS score from coverage alone ──────────────────────────────────────────

def test_the_ats_score_is_eighty_percent_must_haves_and_twenty_percent_preferred():
    assert line_match._coverage_score_out_of_100(0.75, 0.5) == 70
    assert line_match._coverage_score_out_of_100(1.0, 1.0) == 100
    assert line_match._coverage_score_out_of_100(0.0, 1.0) == 20


def test_a_posting_with_only_one_kind_of_line_is_scored_on_that_kind_alone():
    assert line_match._coverage_score_out_of_100(0.6, None) == 60
    assert line_match._coverage_score_out_of_100(None, 0.4) == 40
    assert line_match._coverage_score_out_of_100(None, None) is None


@pytest.mark.asyncio
async def test_every_line_counts_the_same_in_the_ats_score(monkeypatch):
    # Coverage: Kubernetes all of it, the cloud line half, the certification none; Terraform a quarter.
    _jev_sorting_skills(monkeypatch, coverage={"Kubernetes": 4, "Cloud infrastructure (Azure preferred)": 2, "Terraform": 1})

    result = await line_match.ats_score("jev", "Ran Kubernetes clusters.", SKILLS_POSTING)

    assert result["must_have_coverage"] == 0.5
    assert result["preferred_coverage"] == 0.25
    assert result["ats_score"] == round(100 * (0.8 * 0.5 + 0.2 * 0.25))
    assert result["same_occupation"] == 1.0


@pytest.mark.asyncio
async def test_a_career_in_a_neighbouring_occupation_keeps_part_of_the_coverage_score(monkeypatch):
    _jev_sorting_skills(monkeypatch, occupation=2,
                        coverage={"Kubernetes": 4, "Cloud infrastructure (Azure preferred)": 2, "Terraform": 1})

    result = await line_match.ats_score("jev", "Built data pipelines; ran them on Kubernetes.", SKILLS_POSTING)

    assert result["coverage_score"] == 45
    assert result["same_occupation"] == 0.4
    assert result["ats_score"] == 18


@pytest.mark.asyncio
async def test_a_career_in_a_different_occupation_scores_zero_whatever_it_covers(monkeypatch):
    _jev_sorting_skills(monkeypatch, occupation=0, coverage={"Kubernetes": 4})

    result = await line_match.ats_score("jev", "Sold Kubernetes training courses.", SKILLS_POSTING)

    assert result["ats_score"] == 0
    assert result["coverage_score"] > 0


def test_only_the_lower_occupation_levels_take_much_of_the_score_away():
    shares = [share for _description, share in line_match.OCCUPATION_LEVELS]

    assert shares == [0.0, 0.2, 0.4, 0.8, 1.0, 1.0]


@pytest.mark.asyncio
async def test_a_more_important_line_counts_for_more_in_the_ats_score(monkeypatch):
    # Kubernetes is central (counts 4) and fully covered; the other two must-haves count 1 and are not covered.
    _jev_sorting_skills(monkeypatch, coverage={"Kubernetes": 4}, importance={"Kubernetes": 3})

    result = await line_match.ats_score("jev", "Ran Kubernetes clusters.", SKILLS_POSTING)

    kubernetes = [row for row in result["must_have"] if row["line"] == "Kubernetes"][0]
    assert kubernetes["importance"] == 4.0
    assert result["must_have_coverage"] == pytest.approx(4 / 6)


def test_importance_is_asked_about_the_posting_alone():
    assert "resume" not in line_match.IMPORTANCE_QUESTION.lower()
    assert [counts_for for _description, counts_for in line_match.IMPORTANCE_LEVELS] == [1.0, 2.0, 3.0, 4.0]
