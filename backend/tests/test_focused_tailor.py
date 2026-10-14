"""Focused tailoring: a short résumé rebuilt for one posting that says less than the base and never more."""
from backend.analyzer import focused_tailor as focused

BASE = {
    "header": {"name": "Sam Lee", "title": "Backend Developer",
               "contact_items": [{"text": "sam@example.com", "url": "mailto:sam@example.com"}]},
    "summary": "Backend developer with 7+ years of experience in Python and SQL Server.",
    "skills": {"Languages": "Python, JavaScript, SQL", "Databases": "PostgreSQL, SQL Server, Redis",
               "Tools": "GitHub, Jenkins, TDD, Jira"},
    "experience": [
        {"title": "Senior Developer", "company": "", "start": "2021-01", "end": "", "current": True, "bullets": [
            "Built REST APIs with Python and FastAPI for 40 internal teams.",
            "Cut report run time by 30% by tuning PostgreSQL queries.",
            "Wrote unit tests and followed TDD on every service.",
            "Organised the team's yearly offsite.",
        ]},
        {"title": "Developer", "company": "", "start": "2017-03", "end": "2020-12", "current": False, "bullets": [
            "Maintained a SQL Server reporting database.",
            "Built deployment jobs in Jenkins.",
            "Supported a Redis cache used by the web team.",
        ]},
    ],
    "projects": [], "education": [{"degree": "BSc Computer Science", "institution": "State University"}],
    "certifications": [],
}
# Four roles with more bullets than a focused copy keeps: eight, six, five and five.
LONG = {**BASE, "experience": [
    {"title": f"Developer {role}", "company": "", "start": f"20{24 - 2 * role}-01", "end": f"20{25 - 2 * role}-12", "current": False,
     "bullets": [f"Built service {n} with Python." for n in range(1, count + 1)]}
    for role, count in ((1, 8), (2, 6), (3, 5), (4, 5))]}

def _long_with(number: int, text: str) -> dict:
    """LONG with one bullet of its first role replaced."""
    first = LONG["experience"][0]
    bullets = [text if n == number else bullet for n, bullet in enumerate(first["bullets"], start=1)]
    return {**LONG, "experience": [{**first, "bullets": bullets}] + LONG["experience"][1:]}


POSTING = ("Senior Python Engineer at Acme\n\nWe are seeking a Python Developer.\n"
           "Required: Python, FastAPI, POSTGRES, MSSQL, Test Driven Development, Github Actions, Javascript ES6+, Kubernetes.")


def _bullet(text, *sources):
    return {"text": text, "from_bullets": list(sources)}


def _reply(**changes):
    reply = {
        "target_job_title": "Senior Python Engineer",
        "requirements": [], "same_thing_other_name": [],
        "professional_summary": "Backend developer with 7+ years of experience in Python.",
        "technical_skills": [{"category": "Core", "skills": "Python, FastAPI"}],
        "work_experience": [
            {"role_number": 1, "bullets": [_bullet("Built REST APIs with Python and FastAPI for 40 internal teams.", 1),
                                           _bullet("Applied TDD and unit tests on every service.", 3)]},
            {"role_number": 2, "bullets": [_bullet("Maintained a SQL Server reporting database.", 1),
                                           _bullet("Built deployment jobs in Jenkins.", 2)]},
        ],
        "projects": [], "certifications": [],
    }
    reply.update(changes)
    return reply


def _bullets(copy, role_number):
    return copy["experience"][role_number - 1]["bullets"]


# ── What reaches the model ─────────────────────────────────────────────────────

def test_the_prompt_numbers_every_role_and_bullet():
    text = focused.resume_as_numbered_text(BASE)
    assert "ROLE 1: Senior Developer" in text and "ROLE 2: Developer" in text
    assert "  4. Organised the team's yearly offsite." in text


# ── Facts come from the base ───────────────────────────────────────────────────

def test_every_role_stays_with_its_own_title_and_dates():
    copy = focused.build_focused_resume(_reply(work_experience=[]), BASE, POSTING)
    assert [role["title"] for role in copy["experience"]] == ["Senior Developer", "Developer"]
    assert copy["experience"][1]["start"] == "2017-03"


def test_a_role_the_reply_left_empty_keeps_its_first_bullets():
    copy = focused.build_focused_resume(_reply(work_experience=[]), BASE, POSTING)
    assert _bullets(copy, 2) == BASE["experience"][1]["bullets"]   # it has three, and a second role may have four


def test_roles_with_no_employer_name_are_told_apart_by_their_number():
    copy = focused.build_focused_resume(_reply(), BASE, POSTING)
    assert "Jenkins" in " ".join(_bullets(copy, 2)) and "Jenkins" not in " ".join(_bullets(copy, 1))


def test_the_postings_title_goes_on_top_in_place_of_the_own_headline():
    header = focused.build_focused_resume(_reply(), BASE, POSTING)["header"]
    assert header["target_title"] == "Senior Python Engineer" and header["title"] == ""


def test_a_title_the_posting_does_not_state_is_refused_and_the_own_headline_stays():
    header = focused.build_focused_resume(_reply(target_job_title="Staff Engineer"), BASE, POSTING)["header"]
    assert "target_title" not in header and header["title"] == "Backend Developer"


def test_the_work_authorization_the_contact_line_prints_is_kept():
    base = {**BASE, "header": {**BASE["header"], "contact_items": BASE["header"]["contact_items"] + [{"text": "H 1B", "url": ""}]}}
    copy = focused.build_focused_resume(_reply(header={"location": "", "work_authorization": "H 1B"}), base, POSTING)
    assert [item["text"] for item in copy["header"]["contact_items"]] == ["sam@example.com", "H 1B"]


def test_a_work_authorization_the_resume_does_not_print_is_not_added():
    copy = focused.build_focused_resume(_reply(header={"location": "", "work_authorization": "US Citizen"}), BASE, POSTING)
    assert [item["text"] for item in copy["header"]["contact_items"]] == ["sam@example.com"]


def test_education_is_copied_as_the_base_has_it():
    copy = focused.build_focused_resume(_reply(), BASE, POSTING)
    assert copy["education"] == [{"degree": "BSc Computer Science", "institution": "State University"}]


# ── Bullets say no more than their role ────────────────────────────────────────

def test_dropped_bullets_stay_dropped():
    roles = [{"role_number": 1, "bullets": [_bullet(f"Built service {n} in Python.", n) for n in range(1, 6)]}]
    copy = focused.build_focused_resume(_reply(work_experience=roles), LONG, POSTING)
    assert _bullets(copy, 1) == [f"Built service {n} in Python." for n in range(1, 6)]


def test_a_reworded_bullet_is_kept():
    copy = focused.build_focused_resume(_reply(), BASE, POSTING)
    assert "Applied TDD and unit tests on every service." in _bullets(copy, 1)


def test_a_bullet_naming_a_tool_its_role_never_names_goes_back_to_the_original_sentence():
    roles = [{"role_number": 1, "bullets": [_bullet("Built REST APIs with Python and FastAPI on Kubernetes.", 1),
                                            _bullet("Applied TDD on every service.", 3)]}]
    copy = focused.build_focused_resume(_reply(work_experience=roles), BASE, POSTING)
    assert _bullets(copy, 1)[0] == "Built REST APIs with Python and FastAPI for 40 internal teams."


def test_a_tool_from_another_role_does_not_count():
    roles = [{"role_number": 1, "bullets": [_bullet("Built REST APIs with Python, FastAPI and Jenkins.", 1),
                                            _bullet("Applied TDD on every service.", 3)]}]
    copy = focused.build_focused_resume(_reply(work_experience=roles), BASE, POSTING)
    assert "Jenkins" not in " ".join(_bullets(copy, 1))


def test_a_bullet_with_a_number_its_role_never_states_goes_back_to_the_original_sentence():
    roles = [{"role_number": 1, "bullets": [_bullet("Cut report run time by 45% by tuning PostgreSQL queries.", 2),
                                            _bullet("Applied TDD on every service.", 3)]}]
    copy = focused.build_focused_resume(_reply(work_experience=roles), BASE, POSTING)
    assert _bullets(copy, 1)[0] == "Cut report run time by 30% by tuning PostgreSQL queries."


def test_a_bullet_that_names_no_source_and_resembles_none_is_left_out():
    roles = [{"role_number": 1, "bullets": [_bullet("Led a company-wide security audit."),
                                            _bullet("Built REST APIs with Python and FastAPI.", 1),
                                            _bullet("Applied TDD on every service.", 3)]}]
    copy = focused.build_focused_resume(_reply(work_experience=roles), BASE, POSTING)
    assert not any("security audit" in bullet for bullet in _bullets(copy, 1))


def test_an_acronym_the_role_states_may_be_written_out_in_the_postings_words():
    roles = [{"role_number": 1, "bullets": [_bullet("Followed Test Driven Development on every service.", 3),
                                            _bullet("Built REST APIs with Python and FastAPI.", 1)]}]
    copy = focused.build_focused_resume(_reply(work_experience=roles), BASE, POSTING)
    assert "Followed Test Driven Development on every service." in _bullets(copy, 1)


# ── The posting's name for the same thing ──────────────────────────────────────

def test_the_postings_name_is_allowed_for_the_same_thing():
    names = [{"posting_name": "MSSQL", "resume_name": "SQL Server"}]
    roles = [{"role_number": 2, "bullets": [_bullet("Maintained an MSSQL reporting database.", 1),
                                            _bullet("Built deployment jobs in Jenkins.", 2)]}]
    copy = focused.build_focused_resume(_reply(same_thing_other_name=names, work_experience=roles), BASE, POSTING)
    assert "Maintained an MSSQL reporting database." in _bullets(copy, 2)


def test_the_resumes_name_with_words_added_is_a_narrower_thing_and_is_refused():
    names = [{"posting_name": "Github Actions", "resume_name": "GitHub"}]
    assert focused.other_names_allowed({"same_thing_other_name": names}, "GitHub and Jenkins", POSTING) == {}


def test_another_name_counts_only_when_the_resume_states_its_own():
    names = [{"posting_name": "Kubernetes", "resume_name": "Docker Swarm"}]
    assert focused.other_names_allowed({"same_thing_other_name": names}, "Python and Jenkins", POSTING) == {}


# ── Skills ─────────────────────────────────────────────────────────────────────

def _skills_printed(copy):
    return [skill.strip() for value in copy["skills"].values() for skill in value.split(",")]


def test_a_skill_the_resume_does_not_have_is_left_out():
    skills = [{"category": "Core", "skills": "Python, Kubernetes, FastAPI"}]
    copy = focused.build_focused_resume(_reply(technical_skills=skills), BASE, POSTING)
    assert "Kubernetes" not in _skills_printed(copy) and "Python" in _skills_printed(copy)


def test_a_narrower_skill_is_printed_under_the_resumes_own_name():
    skills = [{"category": "Core", "skills": "Python, Javascript ES6+"}]
    copy = focused.build_focused_resume(_reply(technical_skills=skills), BASE, POSTING)
    # the résumé's own skill, in the posting's spelling of that word, without the "ES6+" the résumé never states
    assert "Javascript" in _skills_printed(copy) and "Javascript ES6+" not in _skills_printed(copy)


def test_a_detail_in_brackets_the_resume_does_not_have_is_taken_off():
    skills = [{"category": "Core", "skills": "Python, Jenkins (Github Actions)"}]
    copy = focused.build_focused_resume(_reply(technical_skills=skills), BASE, POSTING)
    assert "Jenkins" in _skills_printed(copy) and not any("Actions" in skill for skill in _skills_printed(copy))


def test_skills_the_reply_dropped_are_not_put_back_unless_the_posting_names_them():
    copy = focused.build_focused_resume(_reply(), BASE, POSTING)
    assert "Jenkins" not in _skills_printed(copy) and "Redis" not in _skills_printed(copy)


def test_a_listed_skill_the_posting_names_is_never_left_out():
    posting = POSTING + " Redis is a plus."
    copy = focused.build_focused_resume(_reply(), BASE, posting)
    assert "Redis" in _skills_printed(copy)


# ── Summary ────────────────────────────────────────────────────────────────────

def test_the_summary_may_be_shorter_than_the_bases():
    copy = focused.build_focused_resume(_reply(), BASE, POSTING)
    assert copy["summary"] == "Backend developer with 7+ years of experience in Python."


def test_a_summary_sentence_stating_what_the_resume_does_not_is_left_out():
    summary = "Backend developer with 7+ years of experience in Python. Expert in Kubernetes and Terraform."
    copy = focused.build_focused_resume(_reply(professional_summary=summary), BASE, POSTING)
    assert copy["summary"] == "Backend developer with 7+ years of experience in Python."


def test_a_summary_with_a_number_the_resume_never_states_loses_that_sentence():
    summary = "Backend developer with 12+ years of experience. Works in Python and FastAPI."
    copy = focused.build_focused_resume(_reply(professional_summary=summary), BASE, POSTING)
    assert copy["summary"] == "Works in Python and FastAPI."


def test_a_short_title_does_not_end_a_sentence():
    summary = "Worked as a Sr. developer on Kubernetes platforms. Works in Python."
    copy = focused.build_focused_resume(_reply(professional_summary=summary), BASE, POSTING)
    assert copy["summary"] == "Works in Python."


# ── Must-haves stay shown, and gaps are reported ───────────────────────────────

def _written(*numbers):
    """A reply for role 1 of LONG that rewrote these bullets, one for one."""
    return [{"role_number": 1, "bullets": [_bullet(f"Built service {n} in Python.", n) for n in numbers]}]


def test_each_role_gets_five_then_four_then_three_bullets():
    copy = focused.build_focused_resume(_reply(work_experience=[]), LONG, POSTING)
    assert [len(role["bullets"]) for role in copy["experience"]] == [5, 4, 3, 3]


def test_a_role_given_too_many_bullets_is_cut_to_its_number():
    copy = focused.build_focused_resume(_reply(work_experience=_written(1, 2, 3, 4, 5, 6, 7)), LONG, POSTING)
    assert len(_bullets(copy, 1)) == 5


def test_a_role_given_too_few_bullets_is_filled_from_the_resume():
    copy = focused.build_focused_resume(_reply(work_experience=_written(3)), LONG, POSTING)
    assert len(_bullets(copy, 1)) == 5 and "Built service 3 in Python." in _bullets(copy, 1)


def test_when_the_reply_wrote_too_many_the_one_that_shows_a_must_is_kept():
    requirements = [{"name": "Redis", "kind": "must", "by": ["1.7"]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 5, 6, 7)), LONG, POSTING)
    assert "Built service 7 in Python." in _bullets(copy, 1) and "Built service 5 in Python." not in _bullets(copy, 1)


def test_a_must_counts_for_more_than_a_duty_when_choosing():
    requirements = [{"name": "Daily stand-ups with the team", "kind": "duty", "by": ["1.6"]}, {"name": "Redis", "kind": "must", "by": ["1.7"]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 6, 7)), LONG, POSTING)
    assert {"Built service 6 in Python.", "Built service 7 in Python."} <= set(_bullets(copy, 1))
    assert "Built service 4 in Python." not in _bullets(copy, 1)


def test_the_replys_own_bullets_are_not_traded_for_the_resumes_when_it_wrote_the_right_number():
    copy = focused.build_focused_resume(_reply(work_experience=_written(4, 5, 6, 7, 8)), LONG, POSTING)
    assert _bullets(copy, 1) == [f"Built service {n} in Python." for n in (4, 5, 6, 7, 8)]


def test_the_bullets_kept_stay_in_the_order_they_stood_in():
    requirements = [{"name": "Redis", "kind": "must", "by": ["1.7"]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 5, 6, 7)), LONG, POSTING)
    assert _bullets(copy, 1) == [f"Built service {n} in Python." for n in (1, 2, 3, 4, 7)]


def test_a_bullet_with_a_measured_result_is_preferred_when_cutting():
    roles = [{"role_number": 1, "bullets": [_bullet(f"Built service {n} in Python.", n) for n in range(1, 6)]
              + [_bullet("Cut run time by 30% in Python.", 6)]}]
    long = _long_with(6, "Cut run time by 30% with Python.")
    copy = focused.build_focused_resume(_reply(work_experience=roles), long, POSTING)
    assert "Cut run time by 30% in Python." in _bullets(copy, 1) and len(_bullets(copy, 1)) == 5


def test_a_must_no_chosen_bullet_shows_is_swapped_in_as_the_resume_wrote_it():
    long = _long_with(8, "Cached results in Redis.")
    requirements = [{"name": "Redis", "kind": "must", "has": True, "by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 5)), long, POSTING)
    assert "Cached results in Redis." in _bullets(copy, 1) and len(_bullets(copy, 1)) == 5
    assert copy[focused.REPORT_KEY]["bullets_swapped_in"] == 1


def test_a_must_already_named_by_a_chosen_bullet_swaps_nothing():
    requirements = [{"name": "Python", "kind": "must", "has": True, "by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 5)), LONG, POSTING)
    assert copy[focused.REPORT_KEY]["bullets_swapped_in"] == 0


def test_a_must_the_resume_shows_nowhere_swaps_nothing():
    requirements = [{"name": "Kubernetes", "kind": "must", "has": False, "by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 5)), LONG, POSTING)
    assert copy[focused.REPORT_KEY]["bullets_swapped_in"] == 0


def test_a_pointer_at_a_bullet_the_resume_does_not_have_is_ignored():
    requirements = [{"name": "Kubernetes", "kind": "must", "by": ["9.1"]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert copy[focused.REPORT_KEY]["not_supported"] == ["Kubernetes"]


def test_requirements_the_resume_does_not_show_are_reported_as_gaps():
    requirements = [{"name": "Kubernetes", "kind": "must", "has": False, "by": []},
                    {"name": "Daily standups", "kind": "duty", "has": False, "by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert copy[focused.REPORT_KEY]["not_supported"] == ["Kubernetes"]


def test_a_requirement_the_resume_has_outside_its_bullets_is_not_a_gap():
    requirements = [{"name": "Bachelor's degree", "kind": "must", "has": True, "by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert copy[focused.REPORT_KEY]["not_supported"] == []


def test_a_reply_that_is_not_an_object_gives_the_base_with_each_roles_first_bullets():
    copy = focused.build_focused_resume(None, BASE, POSTING)
    assert copy["summary"] == BASE["summary"]
    assert {label: value.lower() for label, value in copy["skills"].items()} == \
        {label: value.lower() for label, value in BASE["skills"].items()}
    assert [role["bullets"] for role in copy["experience"]] == [role["bullets"] for role in BASE["experience"]]


# ── The opening verb claims no more than the source ────────────────────────────

def test_a_bullet_that_opens_by_claiming_more_goes_back_to_its_sources_verb():
    roles = [{"role_number": 1, "bullets": [_bullet("Designed REST APIs with Python and FastAPI.", 1),
                                            _bullet("Applied TDD on every service.", 3)]}]
    copy = focused.build_focused_resume(_reply(work_experience=roles), BASE, POSTING)
    assert _bullets(copy, 1)[0] == "Built REST APIs with Python and FastAPI."


def test_a_verb_the_source_itself_uses_is_left_alone():
    assert focused.with_the_verb_its_sources_use("Designed the API layer.", ["Designed and built the API layer."]) == "Designed the API layer."


def test_an_ordinary_change_of_verb_is_left_alone():
    assert focused.with_the_verb_its_sources_use("Developed REST APIs.", ["Built REST APIs."]) == "Developed REST APIs."


# ── The complete copy: tailored as hard, nothing dropped ───────────────────────

def _complete(**changes):
    return focused.build_complete_resume(_reply(**changes), BASE, POSTING)


def test_a_complete_copy_keeps_every_bullet_of_every_role():
    copy = _complete()
    assert [len(role["bullets"]) for role in copy["experience"]] == [4, 3]
    assert "Organised the team's yearly offsite." in _bullets(copy, 1)


def test_rewritten_bullets_come_first_and_the_rest_follow_as_written():
    copy = _complete()
    assert _bullets(copy, 1)[:2] == ["Built REST APIs with Python and FastAPI for 40 internal teams.",
                                     "Applied TDD and unit tests on every service."]
    assert set(_bullets(copy, 1)[2:]) == {"Cut report run time by 30% by tuning PostgreSQL queries.",
                                          "Organised the team's yearly offsite."}


def test_the_rest_follow_in_the_order_the_reply_asks_for():
    roles = [{"role_number": 1, "bullets": [_bullet("Built REST APIs with Python and FastAPI.", 1)], "other_bullets": [4, 2]}]
    copy = _complete(work_experience=roles)
    assert _bullets(copy, 1)[1:] == ["Organised the team's yearly offsite.",
                                     "Cut report run time by 30% by tuning PostgreSQL queries.",
                                     "Wrote unit tests and followed TDD on every service."]


def test_a_bullet_written_from_two_stands_for_both():
    roles = [{"role_number": 1, "bullets": [_bullet("Built REST APIs with Python and FastAPI, with unit tests and TDD.", 1, 3)]}]
    copy = _complete(work_experience=roles)
    assert len(_bullets(copy, 1)) == 3 and "Wrote unit tests and followed TDD on every service." not in _bullets(copy, 1)


def test_a_role_the_reply_never_mentions_keeps_all_its_bullets():
    copy = _complete(work_experience=[])
    assert _bullets(copy, 2) == BASE["experience"][1]["bullets"]


def test_a_complete_copy_still_refuses_a_tool_the_role_never_names():
    roles = [{"role_number": 1, "bullets": [_bullet("Built REST APIs with Python and FastAPI on Kubernetes.", 1)]}]
    copy = _complete(work_experience=roles)
    assert not any("Kubernetes" in bullet for bullet in _bullets(copy, 1))


def test_a_complete_copy_keeps_every_skill_with_the_replys_groups_first():
    copy = _complete()
    printed = _skills_printed(copy)
    assert list(copy["skills"])[0] == "Core" and printed[:2] == ["Python", "FastAPI"]
    # "Javascript" and "Github" are the posting's spellings of two skills the résumé lists
    assert {"Javascript", "SQL", "PostgreSQL", "SQL Server", "Redis", "Github", "Jenkins", "TDD", "Jira"} <= set(printed)


def test_a_complete_copy_adds_no_skill_the_resume_lacks():
    skills = [{"category": "Core", "skills": "Python, Kubernetes, Github Actions"}]
    printed = _skills_printed(_complete(technical_skills=skills))
    assert "Kubernetes" not in printed and "Github Actions" not in printed


def test_a_skill_printed_under_the_postings_name_is_not_printed_twice():
    names = [{"posting_name": "MSSQL", "resume_name": "SQL Server"}]
    skills = [{"category": "Core", "skills": "Python, MSSQL"}]
    printed = _skills_printed(_complete(same_thing_other_name=names, technical_skills=skills))
    assert "MSSQL" in printed and "SQL Server" not in printed


def test_a_complete_copy_keeps_the_own_headline_beside_the_postings_title():
    header = _complete()["header"]
    assert header["target_title"] == "Senior Python Engineer" and header["title"] == "Backend Developer"


def test_the_complete_prompt_asks_for_the_other_bullets_and_the_focused_one_does_not():
    assert "other_bullets" in focused.focused_prompt(BASE, POSTING, keeps_everything=True)
    assert "other_bullets" not in focused.focused_prompt(BASE, POSTING)


# ── Keywords are spelled exactly as the posting spells them ────────────────────

def _requirement(text, has_it=True):
    return {"name": text, "kind": "must", "has": has_it, "by": []}


def test_keywords_are_taken_apart_and_kept_only_as_the_posting_writes_them():
    reply = {"requirements": [_requirement("POSTGRES, MSSQL"), _requirement("Test Driven Development"), _requirement("Docker Swarm"),
                              _requirement("communication skills")]}
    requirements = focused.requirements_matched(reply, BASE["experience"])
    assert focused.keywords_of_the_posting(requirements, POSTING) == ["POSTGRES", "MSSQL", "Test Driven Development"]


def test_a_skill_the_resume_lists_is_a_keyword_when_the_posting_names_it_in_another_spelling():
    assert "Github" in focused.keywords_of_the_posting([], POSTING, BASE)


def test_a_duty_is_not_a_keyword():
    duty = focused.requirements_matched({"requirements": [{"name": "MSSQL", "kind": "duty", "has": True, "by": []}]}, BASE["experience"])
    assert focused.keywords_of_the_posting(duty, POSTING) == []


def test_the_postings_name_is_added_beside_a_skill_printed_under_the_resumes_name():
    skills = focused.with_the_postings_name_beside_the_resumes({"Databases": "PostgreSQL, SQL Server"}, {"MSSQL": "SQL Server"})
    assert skills == {"Databases": "PostgreSQL, SQL Server (MSSQL)"}


def test_a_postings_name_already_printed_is_not_added_again():
    skills = focused.with_the_postings_name_beside_the_resumes({"Databases": "PostgreSQL, MSSQL"}, {"MSSQL": "SQL Server"})
    assert skills == {"Databases": "PostgreSQL, MSSQL"}


def test_a_keyword_the_copy_holds_is_respelled_as_the_posting_spells_it():
    assert focused.in_the_postings_spelling("Used GITHUB and test-driven development.", ["Github", "Test Driven Development"]) \
        == "Used Github and Test Driven Development."


def test_a_keyword_joined_another_way_is_respelled():
    assert focused.in_the_postings_spelling("Built CI/CD pipelines.", ["CI / CD"]) == "Built CI / CD pipelines."


def test_respelling_never_touches_a_longer_word_that_contains_the_keyword():
    assert focused.in_the_postings_spelling("Tuned PostgreSQL and MySQL.", ["SQL"]) == "Tuned PostgreSQL and MySQL."


def test_the_copy_spells_a_keyword_as_the_posting_does_in_skills_summary_and_bullets():
    posting = POSTING.replace("Python, FastAPI", "PYTHON, FastAPI")
    copy = focused.build_complete_resume(_reply(requirements=[_requirement("PYTHON")]), BASE, posting)
    assert "PYTHON" in _skills_printed(copy) and "Python" not in copy["summary"]
    assert not any("Python" in bullet for role in copy["experience"] for bullet in role["bullets"])


def test_the_report_counts_each_keyword_in_the_skills_and_elsewhere():
    copy = focused.build_complete_resume(_reply(requirements=[_requirement("FastAPI"), _requirement("Kubernetes", False)]), BASE, POSTING)
    counted = {row["keyword"]: row for row in copy[focused.REPORT_KEY]["keywords"]}
    assert counted["FastAPI"]["in_skills"] and counted["FastAPI"]["elsewhere"] >= 1
    assert not counted["Kubernetes"]["in_skills"] and counted["Kubernetes"]["elsewhere"] == 0


# ── Checks that used to refuse too much ────────────────────────────────────────

def test_a_keyword_phrase_is_supported_when_the_role_writes_it_with_another_ending():
    base = {**BASE, "experience": [{**BASE["experience"][0], "bullets": ["Built RESTful APIs with Python and FastAPI.", "Followed TDD."]}]}
    roles = [{"role_number": 1, "bullets": [_bullet("Built REST APIs with Python and FastAPI.", 1), _bullet("Followed TDD.", 2)]}]
    base["skills"] = {**BASE["skills"], "Web": "REST APIs"}
    copy = focused.build_focused_resume(_reply(work_experience=roles), base, POSTING)
    assert _bullets(copy, 1)[0] == "Built REST APIs with Python and FastAPI."


def test_a_short_plain_word_is_not_taken_for_a_skills_initials():
    assert "as" not in focused._names_in("Built APIs as a service", {"as", "apis"})


def test_one_unsupported_item_is_taken_out_of_a_summary_list_and_the_sentence_stays():
    summary = "Backend developer with 7+ years of experience. Works in Python, Kubernetes, and SQL Server."
    copy = focused.build_focused_resume(_reply(professional_summary=summary), BASE, POSTING)
    assert copy["summary"] == "Backend developer with 7+ years of experience. Works in Python, and SQL Server."


def test_a_summary_sentence_that_is_not_a_list_is_still_left_out_whole():
    summary = "Backend developer with 7+ years of experience. Expert in Kubernetes platforms."
    copy = focused.build_focused_resume(_reply(professional_summary=summary), BASE, POSTING)
    assert copy["summary"] == "Backend developer with 7+ years of experience."


def test_a_bullet_that_shows_a_requirement_is_written_as_role_dot_bullet():
    matched = focused.requirements_matched({"requirements": [{"name": "POSTGRES", "kind": "must", "has": True, "by": ["1.2", "9.1", "x"]}]},
                                           BASE["experience"])
    assert matched == [{"requirement": "POSTGRES", "importance": "must", "shown_by": [(1, 2)], "resume_has_it": True}]


def test_a_bullet_that_names_a_must_is_credited_with_it_without_a_pointer():
    requirements = [{"name": "FastAPI", "kind": "must", "by": []}]
    roles = [{"role_number": 1, "bullets": [_bullet(f"Built service {n} in Python.", n) for n in range(1, 6)]
              + [_bullet("Built service 6 with Python and FastAPI.", 6)]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=roles), _long_with(6, "Built service 6 with Python and FastAPI."), POSTING)
    assert "Built service 6 with Python and FastAPI." in _bullets(copy, 1) and len(_bullets(copy, 1)) == 5


def test_a_phrase_is_supported_when_the_bullets_own_sources_state_each_of_its_words():
    assert focused._every_word_is_stated_in("REST APIs", "Built RESTful services and public APIs.")
    assert not focused._every_word_is_stated_in("Spring Security", "Built Spring Boot services.")


def test_a_swap_is_made_only_when_the_copy_then_shows_more():
    long = _long_with(8, "Cached results in Redis.")
    requirements = [{"name": "Redis", "kind": "must", "has": True, "by": []}]
    roles = [{"role_number": 1, "bullets": [_bullet(f"Built service {n} in Python.", n) for n in range(1, 5)] + [_bullet("Cached results in Redis.", 8)]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=roles), long, POSTING)
    assert copy[focused.REPORT_KEY]["bullets_swapped_in"] == 0


def test_a_roles_opening_bullets_are_never_swapped_out():
    long = _long_with(8, "Cached results in Redis.")
    requirements = [{"name": "Redis", "kind": "must", "has": True, "by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 5)), long, POSTING)
    assert _bullets(copy, 1)[:2] == ["Built service 1 in Python.", "Built service 2 in Python."]


def test_no_more_than_the_allowed_number_of_bullets_is_swapped_in_a_role():
    first = LONG["experience"][0]
    bullets = first["bullets"][:5] + ["Used Redis.", "Used Kafka.", "Used Docker."]
    long = {**LONG, "experience": [{**first, "bullets": bullets}] + LONG["experience"][1:]}
    # nice-to-haves: only a must may take more swaps than this
    requirements = [{"name": name, "kind": "nice", "has": True, "by": []} for name in ("Redis", "Kafka", "Docker")]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 5)), long, POSTING)
    assert copy[focused.REPORT_KEY]["bullets_swapped_in"] == focused.MOST_SWAPS_IN_A_ROLE


def test_a_bullet_naming_one_keyword_of_a_longer_requirement_is_credited_with_it():
    assert focused._names_the_requirement("Built screens in React.js.", "React.js and modern frontend development practices")
    assert not focused._names_the_requirement("Built screens in Angular.", "React.js and modern frontend development practices")


def test_one_unsupported_item_is_taken_out_of_a_bullets_list_and_the_rewrite_stays():
    roles = [{"role_number": 1, "bullets": [_bullet("Built REST APIs with Python, FastAPI, Kubernetes, and Docker for 40 internal teams.", 1)]}]
    base = {**BASE, "experience": [{**BASE["experience"][0], "bullets": ["Built REST APIs with Python, FastAPI and Docker for 40 internal teams."]}]}
    copy = focused.build_focused_resume(_reply(work_experience=roles), base, POSTING)
    assert _bullets(copy, 1) == ["Built REST APIs with Python, FastAPI, and Docker for 40 internal teams."]


def test_a_requirement_naming_several_keywords_is_taken_apart():
    matched = focused.requirements_matched({"requirements": [{"name": "Python, Pandas, and FastAPI", "kind": "must", "has": True, "by": ["1.1"]}]},
                                           BASE["experience"])
    assert [m["requirement"] for m in matched] == ["Python", "Pandas", "FastAPI"]
    assert all(m["importance"] == "must" and m["shown_by"] == [(1, 1)] for m in matched)


def test_a_requirement_that_is_a_sentence_stays_whole():
    matched = focused.requirements_matched({"requirements": [{"name": "Collaborate with product managers, architects and QA teams", "kind": "duty", "has": True, "by": []}]},
                                           BASE["experience"])
    assert [m["requirement"] for m in matched] == ["Collaborate with product managers, architects and QA teams"]


def test_every_must_the_resume_names_in_a_bullet_ends_up_in_a_chosen_bullet():
    first = LONG["experience"][0]
    bullets = first["bullets"][:5] + ["Used Redis.", "Used Kafka.", "Used Docker."]
    long = {**LONG, "experience": [{**first, "bullets": bullets}] + LONG["experience"][1:]}
    requirements = [{"name": "Redis, Kafka, Docker", "kind": "must", "has": True, "by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements, work_experience=_written(1, 2, 3, 4, 5)), long, POSTING)
    assert {"Used Redis.", "Used Kafka.", "Used Docker."} <= set(_bullets(copy, 1)) and len(_bullets(copy, 1)) == 5


def test_a_clause_is_never_cut_out_of_a_sentence_as_if_it_were_a_list_item():
    sentence = "Built apps using Java, Spring Boot, and Kubernetes clusters for patient and provider workflows, and billing."
    assert focused._without_the_list_items_naming(sentence, ["Kubernetes"]) == ""


def test_a_short_list_item_is_still_taken_out():
    assert focused._without_the_list_items_naming("Built apps using Java, Kubernetes, and Spring Boot.", ["Kubernetes"]) \
        == "Built apps using Java, and Spring Boot."


def test_an_item_that_carries_more_than_the_name_is_not_cut():
    sentence = "Built apps using Java, Spring Boot, and microservices architecture for patient, provider, and billing workflows."
    assert focused._without_the_list_items_naming(sentence, ["microservices architecture"]) == ""


def test_the_item_that_closes_a_list_in_mid_sentence_is_not_cut():
    sentence = "Built apps using Java, Spring Boot, and Kubernetes, supporting patient, provider, and clinical workflows."
    assert focused._without_the_list_items_naming(sentence, ["Kubernetes"]) == ""


def test_a_section_heading_offered_as_the_title_is_refused():
    posting = "Summary\nWe are looking for a Senior Python Backend Engineer.\nRequired: Python."
    header = focused.build_focused_resume(_reply(target_job_title="Summary"), BASE, posting)["header"]
    assert "target_title" not in header and header["title"] == "Backend Developer"


def test_a_title_named_in_the_text_is_accepted():
    posting = "Summary\nWe are looking for a Senior Python Backend Engineer.\nRequired: Python."
    header = focused.build_focused_resume(_reply(target_job_title="Senior Python Backend Engineer"), BASE, posting)["header"]
    assert header["target_title"] == "Senior Python Backend Engineer"


def test_a_complete_copy_does_not_print_again_a_bullet_a_rewrite_already_says():
    roles = [{"role_number": 1, "bullets": [_bullet("Followed TDD and wrote unit tests on every service.", 1)]}]   # the wrong source
    copy = _complete(work_experience=roles)
    assert "Wrote unit tests and followed TDD on every service." not in _bullets(copy, 1)
    assert "Built REST APIs with Python and FastAPI for 40 internal teams." not in _bullets(copy, 1)[:1]


def test_a_different_bullet_is_still_printed_in_a_complete_copy():
    copy = _complete(work_experience=[{"role_number": 1, "bullets": [_bullet("Followed TDD and wrote unit tests on every service.", 3)]}])
    assert "Organised the team's yearly offsite." in _bullets(copy, 1) and len(_bullets(copy, 1)) == 4
