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
    assert _bullets(copy, 2) == BASE["experience"][1]["bullets"][:focused.FEWEST_BULLETS_OF_A_ROLE]


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
    copy = focused.build_focused_resume(_reply(), BASE, POSTING)
    assert "Organised the team's yearly offsite." not in _bullets(copy, 1)


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
    assert "JavaScript" in _skills_printed(copy) and "Javascript ES6+" not in _skills_printed(copy)


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

def test_a_must_have_the_chosen_bullets_no_longer_show_gets_its_best_bullet_back():
    requirements = [{"requirement": "POSTGRES", "importance": "must_have", "shown_by": [{"role_number": 1, "bullet_number": 2}]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert "Cut report run time by 30% by tuning PostgreSQL queries." in _bullets(copy, 1)
    assert copy[focused.REPORT_KEY]["bullets_put_back_for_a_must_have"] == 1


def test_a_must_have_already_shown_adds_nothing():
    requirements = [{"requirement": "FastAPI", "importance": "must_have", "shown_by": [{"role_number": 1, "bullet_number": 1}]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert len(_bullets(copy, 1)) == 2


def test_a_nice_to_have_does_not_bring_a_bullet_back():
    requirements = [{"requirement": "POSTGRES", "importance": "nice_to_have", "shown_by": [{"role_number": 1, "bullet_number": 2}]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert len(_bullets(copy, 1)) == 2


def test_a_pointer_at_a_bullet_the_resume_does_not_have_is_ignored():
    requirements = [{"requirement": "Kubernetes", "importance": "must_have", "shown_by": [{"role_number": 9, "bullet_number": 1}]}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert copy[focused.REPORT_KEY]["not_supported"] == ["Kubernetes"]


def test_requirements_the_resume_does_not_show_are_reported_as_gaps():
    requirements = [{"requirement": "Kubernetes", "importance": "must_have", "resume_has_it": False, "shown_by": []},
                    {"requirement": "Daily standups", "importance": "duty", "resume_has_it": False, "shown_by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert copy[focused.REPORT_KEY]["not_supported"] == ["Kubernetes"]


def test_a_requirement_the_resume_has_outside_its_bullets_is_not_a_gap():
    requirements = [{"requirement": "Bachelor's degree", "importance": "must_have", "resume_has_it": True, "shown_by": []}]
    copy = focused.build_focused_resume(_reply(requirements=requirements), BASE, POSTING)
    assert copy[focused.REPORT_KEY]["not_supported"] == []


def test_a_reply_that_is_not_an_object_gives_the_base_cut_to_its_first_bullets():
    copy = focused.build_focused_resume(None, BASE, POSTING)
    assert copy["summary"] == BASE["summary"] and copy["skills"] == BASE["skills"]
    assert all(len(role["bullets"]) == focused.FEWEST_BULLETS_OF_A_ROLE for role in copy["experience"])
