"""Tailoring answers in the ATS knowledge base's sections and order, through the ladder, and never lets the model change a fact."""
import json
import re
import uuid
from pathlib import Path

import pytest

from backend.analyzer.resume_schema import (ATS_LAYOUT, LAYOUT_KEY, TAILOR_JSON_SCHEMA,
                                            TAILOR_SCHEMA_NAME, build_base_resume,
                                            build_tailored_resume,
                                            unsupported_claims)
from backend.models.db import Job, Resume, Setting

KB_SECTIONS = ["header", "professional_summary", "technical_skills", "work_experience",
               "projects", "education", "certifications"]

BASE = {
    "header": {
        "name": "Shah Ahsan Ali",
        "title": "Senior Data Engineer",
        "contact_items": [
            {"text": "Nashville, TN"},
            {"text": "shah@example.com", "url": "mailto:shah@example.com"},
            {"text": "+1 401-206-0657"},
            {"text": "U.S. Citizen"},
            {"text": "LinkedIn", "url": "https://linkedin.com/in/shah", "stub": "l"},
        ],
    },
    "summary": "8+ years of experience building data platforms.",
    "skills": {"ETL / ELT": "Azure Data Factory, Adobe Airflow",
               "Certifications": "AWS Certified Data Engineer"},
    "experience": [
        {"company": "Matrix Medical Network", "title": "Senior Data Engineer",
         "location": "Nashville, TN", "date": "Oct 2024 - Present",
         "start": "2024-10", "end": "", "current": True, "description": "",
         "bullets": ["Engineered Databricks notebooks, improving processing efficiency by 45%.",
                     "Organised the team offsite."]},
        {"company": "SS&C Technologies", "title": "Data Engineer", "location": "Windsor, CT",
         "date": "12/2022 - 9/2024", "start": "2022-12", "end": "2024-09", "current": False,
         "description": "", "bullets": ["Built ADF pipelines for financial data."]},
    ],
    "education": [{"school": "State University", "location": "Boston, MA",
                   "degree": "Bachelor in Computer Science", "years": "2014 - 2018",
                   "start": "2014", "end": "2018", "current": False}],
    "projects": [{"name": "Side project", "bullets": ["p"]}],
    "publications": [{"title": "A paper"}],
}

REPLY = {
    "header": {"name": "Shah Ali", "location": "Nashville, TN", "email": "other@example.com",
               "phone": "", "work_authorization": "US Citizen", "github": "github.com/invented",
               "linkedin": "https://linkedin.com/in/shah"},
    "professional_summary": "Accomplished Senior Data Engineer with 8+ years of experience.",
    "technical_skills": [{"category": "ETL/ELT", "skills": "Azure Data Factory, Apache Airflow"}],
    # Reordered, and with a title the model "improved" — neither may move bullets across roles.
    "work_experience": [
        {"job_title": "Data Engineer", "company": "SS&C Technologies", "location": "", "dates": "",
         "bullets": ["Built Azure Data Factory pipelines for financial transactions."]},
        {"job_title": "Lead Data Engineer", "company": "Matrix Medical Network", "location": "",
         "dates": "", "bullets": ["Engineered Azure Databricks notebooks, improving processing efficiency by 45%."]},
    ],
    # The base project comes back reworded; an invented one must not.
    "projects": [
        {"name": "Side Project", "url": "", "dates": "", "technologies": "",
         "bullets": ["Built a side project pipeline."]},
        {"name": "Invented Project", "url": "", "dates": "", "technologies": "", "bullets": ["x"]},
    ],
    "education": [{"degree": "PhD", "university": "Invented University", "graduation_year": "2030"}],
    "certifications": ["AWS Certified Data Engineer", "Azure Data Factory", "Made-up Cert Pro"],
}


# ── the response format ─────────────────────────────────────────────────────

def test_schema_sections_are_the_kb_sections_in_order():
    assert list(TAILOR_JSON_SCHEMA["properties"]) == KB_SECTIONS
    assert TAILOR_JSON_SCHEMA["required"] == KB_SECTIONS
    assert list(TAILOR_JSON_SCHEMA["properties"]["header"]["properties"]) == [
        "name", "title", "location", "email", "phone", "work_authorization", "github", "linkedin"]


def test_seeded_prompts_leave_the_shape_to_the_schema_and_never_invite_invention():
    """The response format carries the shape; the prompt carries only what a schema cannot say."""
    from backend.seed import DEFAULT_SETTINGS
    for key in ("cv_tailor_prompt", "persona_tailor_prompt"):
        prompt = DEFAULT_SETTINGS[key][0]
        assert "{resume_json}" in prompt and "{job_description}" in prompt
        assert "Stick to the strict JSON format" in prompt
        # no second copy of the schema: its field names never appear as JSON keys
        for section in KB_SECTIONS:
            assert f'"{section}"' not in prompt
        assert "suggested_bullets" not in prompt
        assert "MAY invent" not in prompt
        assert "no maximum" in prompt and "counts as work not done" in prompt


# ── folding the reply onto the base ─────────────────────────────────────────

def test_the_copy_holds_only_the_kb_sections_in_order():
    out = build_tailored_resume(REPLY, BASE)
    assert [k for k in out if not k.startswith("_")] == [
        "header", "summary", "skills", "experience", "projects", "education", "certifications"]
    assert out[LAYOUT_KEY] == ATS_LAYOUT
    assert "publications" not in out


def test_projects_keep_base_facts_and_never_gain_an_invented_one():
    projects = build_tailored_resume(REPLY, BASE)["projects"]
    assert projects == [{**BASE["projects"][0], "bullets": ["Built a side project pipeline."]}]


def test_a_project_the_reply_omits_keeps_its_base_bullets():
    """Silence never loses a project: models drop one by omission, so dropping takes saying so."""
    base = {**BASE, "projects": [BASE["projects"][0], {"name": "Second project", "bullets": ["kept"]}]}
    out = build_tailored_resume(REPLY, base)
    assert [p["name"] for p in out["projects"]] == ["Side project", "Second project"]
    assert out["projects"][1]["bullets"] == ["kept"]


def test_a_project_returned_with_no_bullets_is_dropped():
    reply = {**REPLY, "projects": [{"name": "Side Project", "url": "", "dates": "",
                                    "technologies": "", "bullets": []}]}
    assert build_tailored_resume(reply, BASE)["projects"] == []


def test_a_reply_without_projects_keeps_the_base_projects():
    reply = {k: v for k, v in REPLY.items() if k != "projects"}
    assert build_tailored_resume(reply, BASE)["projects"] == BASE["projects"]


def test_bullets_follow_their_own_role_even_when_the_reply_reorders_roles():
    exp = build_tailored_resume(REPLY, BASE)["experience"]
    assert exp[0]["company"] == "Matrix Medical Network"
    assert exp[0]["bullets"] == ["Engineered Azure Databricks notebooks, improving processing efficiency by 45%."]
    assert exp[1]["company"] == "SS&C Technologies"
    assert exp[1]["bullets"] == ["Built Azure Data Factory pipelines for financial transactions."]


def test_a_single_date_prints_alone_as_the_end_date():
    """One printed date means the role ended then: "Jun 2025", never a dangling dash."""
    from backend.analyzer.resume_schema import _role_dates
    # unchanged: a real range, and an ongoing role
    assert _role_dates({"start": "2021-05", "end": "2025-09", "current": False}) == "May 2021 – Sep 2025"
    assert _role_dates({"start": "2024-10", "end": "", "current": True}) == "Oct 2024 – Present"
    # a lone date stands on its own, whichever field it landed in
    assert _role_dates({"start": "", "end": "2025-06", "current": False}) == "Jun 2025"
    assert _role_dates({"start": "2021-05", "end": "", "current": False}) == "May 2021"
    assert _role_dates({"start": "", "end": "2025", "current": False}) == "2025"
    # nothing at all still falls back to whatever the résumé printed
    assert _role_dates({"start": "", "end": "", "current": False}) == ""
    for out in (_role_dates({"start": "", "end": "2025-06", "current": False}),
                _role_dates({"start": "2021-05", "end": "", "current": False})):
        assert "–" not in out, out

    base = {**BASE, "experience": [{**BASE["experience"][0],
                                    "start": "", "end": "2025-06", "current": False}]}
    reply = {**REPLY, "work_experience": [{"job_title": "Senior Data Engineer",
                                           "company": "Matrix Medical Network", "location": "",
                                           "dates": "", "bullets": ["kept"]}]}
    assert build_tailored_resume(reply, base, "")["experience"][0]["date"] == "Jun 2025"


def test_structuring_rules_say_a_lone_date_is_the_end():
    from backend.api.routes_resumes import _BASE_RULES
    assert "only ONE date has ended on it" in _BASE_RULES
    assert "graduation date is the END" in _BASE_RULES


def test_role_facts_come_from_the_base_and_dates_read_month_year():
    exp = build_tailored_resume(REPLY, BASE)["experience"]
    assert exp[0]["title"] == "Senior Data Engineer"          # not the reply's "Lead"
    assert exp[0]["location"] == "Nashville, TN"
    assert exp[0]["date"] == "Oct 2024 – Present"
    assert exp[1]["date"] == "Dec 2022 – Sep 2024"


def test_a_role_the_reply_skipped_keeps_its_base_bullets():
    reply = {**REPLY, "work_experience": REPLY["work_experience"][:1]}
    exp = build_tailored_resume(reply, BASE)["experience"]
    assert exp[0]["bullets"] == BASE["experience"][0]["bullets"]


def test_header_is_the_base_items_in_kb_order_and_nothing_invented():
    header = build_tailored_resume(REPLY, BASE)["header"]
    assert header["name"] == "Shah Ahsan Ali"
    assert header["title"] == "Senior Data Engineer"   # the base's headline, kept
    texts = [i["text"] for i in header["contact_items"]]
    # Location leads, then email, phone, work authorization and the profile links.
    assert texts == ["Nashville, TN", "shah@example.com", "+1 401-206-0657", "U.S. Citizen", "LinkedIn"]
    assert header["contact_items"][-1]["stub"] == "l"          # tracer stub survives
    assert not any("github" in json.dumps(i) for i in header["contact_items"])


def test_location_is_the_base_item_and_never_the_postings_city():
    """The reply points at a location; the base's own item is what prints."""
    moved = build_tailored_resume({**REPLY, "header": {**REPLY["header"], "location": "Austin, TX"}}, BASE)
    texts = [i["text"] for i in moved["header"]["contact_items"]]
    assert "Austin, TX" not in texts
    assert texts[0] == "shah@example.com"        # no grounded location, so none prints

    # A base that split the location out of its contact line still prints it.
    split = {**BASE, "header": {**BASE["header"],
                                "contact_items": BASE["header"]["contact_items"][1:],
                                "contact": {"location": "Nashville, TN"}}}
    kept = build_tailored_resume(REPLY, split)["header"]["contact_items"]
    assert kept[0]["text"] == "Nashville, TN"


def test_the_headline_may_be_narrowed_but_never_invented():
    """A tailored copy may drop part of the résumé's own headline; it may not write a new one."""
    narrowed = build_tailored_resume({**REPLY, "header": {**REPLY["header"], "title": "Data Engineer"}}, BASE)
    assert narrowed["header"]["title"] == "Data Engineer"
    invented = build_tailored_resume({**REPLY, "header": {**REPLY["header"], "title": "Principal Architect"}}, BASE)
    assert invented["header"]["title"] == "Senior Data Engineer"


def test_education_is_the_base_with_its_graduation_date():
    edu = build_tailored_resume(REPLY, BASE)["education"]
    assert edu == [{**BASE["education"][0], "years": "2018"}]

    # A month the base knows is kept: a bare year loses what a parser reads it by.
    dated = {**BASE, "education": [{**BASE["education"][0], "end": "2018-05"}]}
    assert build_tailored_resume(REPLY, dated)["education"][0]["years"] == "May 2018"


def test_only_certifications_the_base_states_survive():
    assert build_tailored_resume(REPLY, BASE)["certifications"] == ["AWS Certified Data Engineer"]


def test_an_empty_reply_section_keeps_the_base():
    out = build_tailored_resume({"professional_summary": "", "technical_skills": []}, BASE)
    assert out["summary"] == BASE["summary"]
    assert out["skills"] == BASE["skills"]


def test_a_skill_the_reply_drops_comes_back_from_the_base():
    """The reply groups, orders and words the skills; the base decides which exist."""
    reply = {**REPLY, "technical_skills": [{"category": "ETL/ELT", "skills": "Azure Data Factory"}]}
    skills = build_tailored_resume(reply, BASE)["skills"]
    # The reply's own wording and category lead, with the dropped skill put back beside it.
    assert skills["ETL/ELT"] == "Azure Data Factory, Adobe Airflow"
    # A credentials category is the one drop that stands: certifications have their own section.
    assert "Certifications" not in skills

    # A whole category the reply left out returns carrying only what it dropped.
    base = {**BASE, "skills": {"ETL / ELT": "Azure Data Factory", "Languages": "Python, SQL"}}
    out = build_tailored_resume(reply, base)["skills"]
    assert out["Languages"] == "Python, SQL"


def test_a_skill_the_base_never_states_is_taken_out():
    """The base is the ceiling as well as the floor: a JD keyword with nothing behind it goes."""
    reply = {**REPLY, "technical_skills": [
        {"category": "ETL/ELT", "skills": "Azure Data Factory, TCP/IP, network topology"}]}
    skills = build_tailored_resume(reply, BASE)["skills"]
    joined = " ".join(skills.values())
    assert "TCP/IP" not in joined and "network topology" not in joined
    # Rule 5 asks for the posting's spelling, so an acronym the base states keeps it.
    acronyms = {**BASE, "skills": {"ETL / ELT": "ADF"}}
    kept = build_tailored_resume(reply, acronyms)["skills"]
    assert "Azure Data Factory" in " ".join(kept.values())


def test_dropped_source_terms_catches_a_keyword_the_structuring_lost():
    """The document is the pool: a term lost at structuring can never be tailored back in."""
    from backend.analyzer.resume_schema import dropped_source_terms
    source = ("TECHNICAL SKILLS\n"
              "Troubleshooting Tools: Proficient with Ping, Traceroute, and Nslookup for\n"
              "connectivity, latency, and DNS diagnostics.\n")
    lost = {"skills": {"Troubleshooting Tools": "Ping, Traceroute, Nslookup"}}
    assert dropped_source_terms(lost, source) == ["DNS"]

    kept = {"skills": {"Troubleshooting Tools": "Ping, Traceroute, Nslookup, DNS diagnostics"}}
    assert dropped_source_terms(kept, source) == []

    # An all-capitals heading is not a dropped acronym.
    assert "TECHNICAL" not in dropped_source_terms(lost, source)


def test_structuring_rules_keep_terms_named_as_context():
    from backend.api.routes_resumes import _BASE_RULES
    assert "four searchable terms" in _BASE_RULES
    assert "Skills stay in the skills section" in _BASE_RULES


def test_technology_names_are_spelled_the_way_a_posting_spells_them():
    """A strict keyword matcher scores "Fast API" as zero against a JD asking for FastAPI."""
    from backend.analyzer.resume_schema import canonicalize_tech as c
    assert c("Built with Fast API and SQL Alchemy.") == "Built with FastAPI and SQLAlchemy."
    assert c("Wrote Py Test suites and Junit tests.") == "Wrote pytest suites and JUnit tests."
    assert c("Used Java Script and Postgre SQL.") == "Used JavaScript and PostgreSQL."
    assert c("Stored in Fire store and Mongo DB.") == "Stored in Firestore and MongoDB."
    assert c("Charts with Num Py and scikit learn.") == "Charts with NumPy and scikit-learn."

    # Names whose correct spelling KEEPS the space must never be joined.
    for spaced in ("Reports in Power BI from SQL Server.",
                   "Pipelines in Azure Data Factory.",
                   "Models from Hugging Face.",
                   "Built a Web API in ASP.NET.",
                   "Windows Server 2012 and Google Calendar."):
        assert c(spaced) == spaced, spaced

    # Prose is not a product: "a fast API response" is English.
    assert c("Delivered a fast API response.") == "Delivered a fast API response."

    # A link inside a bullet is an address, not prose — recasing it would break it.
    assert c("Published at github.com/foo/fastapi-demo") == "Published at github.com/foo/fastapi-demo"
    assert c("See https://github.com/tiangolo/fastapi for Fast API docs.") == (
        "See https://github.com/tiangolo/fastapi for FastAPI docs.")
    assert c("Shipped CI/CD with Argo CD.") == "Shipped CI/CD with ArgoCD."


def test_a_link_inside_prose_is_never_recased():
    """"github.com/foo/fastapi-demo" is an address; recasing it stops it resolving."""
    from backend.analyzer.resume_schema import canonicalize_tech as c
    for link in ("github.com/ZuhairBaig1",
                 "Published at github.com/foo/fastapi-demo",
                 "Contact fast.api@example.com."):
        assert c(link) == link, link
    # the prose around a link is still repaired
    assert c("See https://github.com/tiangolo/fastapi for Fast API docs.") == (
        "See https://github.com/tiangolo/fastapi for FastAPI docs.")
    # a bare slash is not a link
    assert c("Shipped CI/CD with Argo CD.") == "Shipped CI/CD with ArgoCD."


def test_builders_spell_technology_names_canonically():
    reply = {"professional_summary": "Engineer using Fast API.",
             "technical_skills": [{"category": "Backend", "skills": "Fast API, SQL Alchemy"}],
             "work_experience": [{"job_title": "Data Engineer", "company": "SS&C Technologies",
                                  "location": "", "dates": "", "start": "", "end": "",
                                  "current": False,
                                  "bullets": ["Built services with Fast API and Py Test."]}],
             "projects": [], "education": [], "certifications": [],
             "header": {"name": "Shah", "title": "", "location": "", "email": "", "phone": "",
                        "work_authorization": "", "github": "", "linkedin": ""}}
    base = build_base_resume(reply)
    assert base["summary"] == "Engineer using FastAPI."
    assert base["skills"]["Backend"] == "FastAPI, SQLAlchemy"
    assert base["experience"][0]["bullets"] == ["Built services with FastAPI and pytest."]

    tailored = build_tailored_resume(reply, base)
    assert tailored["experience"][0]["bullets"] == ["Built services with FastAPI and pytest."]


def test_unsupported_claims_flags_a_bullet_the_role_does_not_support():
    """Rule 0 made checkable — reporting only, so the copy itself is untouched."""
    base = {"skills": {"Net": "DHCP, SSH"},
            "experience": [{"company": "PSF", "bullets": ["Configured VLANs and trunk ports."]}]}
    tailored = {"summary": "Engineer with LangGraph experience.",
                "experience": [{"company": "PSF", "bullets": [
                    "Configured VLANs and trunk ports.",          # the base's own, fine
                    "Configured DHCP and SSH on Cisco IOS."]}]}   # built out of the skills list
    flagged = unsupported_claims(tailored, base)
    assert ("summary", "LangGraph") in flagged
    assert {t for where, t in flagged if where.startswith("experience")} >= {"DHCP", "SSH"}
    # A bullet the role does support raises nothing.
    assert not [t for where, t in flagged if t == "VLANs"]


def test_a_rewrite_that_drops_what_the_posting_asks_for_keeps_the_base_summary():
    """The summary lost a third of its words every run, taking the JD's own terms with it."""
    base = {**BASE, "summary": "Engineer using Azure Data Factory, Adobe Airflow and Databricks.",
            "skills": {"ETL / ELT": "Azure Data Factory, Adobe Airflow, Databricks"}}
    jd = "We need Azure Data Factory and Databricks experience on a modern data platform."

    thinned = {**REPLY, "professional_summary": "Engineer using Adobe Airflow."}
    assert build_tailored_resume(thinned, base, jd)["summary"] == base["summary"]

    # A rewrite that keeps them is used as written.
    kept = {**REPLY, "professional_summary": "Azure Data Factory and Databricks engineer."}
    assert build_tailored_resume(kept, base, jd)["summary"] == "Azure Data Factory and Databricks engineer."

    # With no posting text there is nothing to protect, so the rewrite stands.
    assert build_tailored_resume(thinned, base, "")["summary"] == "Engineer using Adobe Airflow."


def test_jd_protected_terms_are_what_the_resume_and_the_posting_share():
    from backend.analyzer.resume_schema import jd_protected_terms
    base = {"skills": {"Databases": "PostgreSQL, Oracle, MongoDB", "Lang": "Java, Python"}}
    shared = jd_protected_terms(base, "Expert in Python and PostgreSQL on a cloud platform.")
    assert set(shared) == {"PostgreSQL", "Python"}


def test_a_bare_name_and_its_detail_group_are_one_skill_and_the_fuller_one_prints():
    """"AWS" and "AWS (EC2, EKS, S3)" are the same skill; printing both read as a bug."""
    from backend.analyzer.resume_schema import _same_skill
    assert _same_skill("AWS", "AWS (EC2, EKS, S3)")
    assert _same_skill("GCP", "GCP (GKE, Compute Engine)")
    assert not _same_skill("AWS (ECS, Lambda)", "Azure (App Services)")
    assert not _same_skill("Helm", "Helm Deployments")      # different things, both kept

    base = {**BASE, "skills": {"Cloud": "AWS (EC2, EKS, S3, RDS), Microsoft Azure (AKS)"}}
    # the reply names the clouds bare AND repeats one with its own detail list
    reply = {**REPLY, "technical_skills": [
        {"category": "Cloud", "skills": "AWS, Microsoft Azure, AWS (EKS, ECS, EC2)"}]}
    line = build_tailored_resume(reply, base, "")["skills"]["Cloud"]

    assert line.count("AWS") == 1, line                      # printed once, not twice
    assert "EC2, EKS, S3, RDS" in line                        # the fuller spelling won
    assert "ECS" not in line                                  # and the invented item went with it


def test_a_spelled_out_acronym_does_not_duplicate_the_skill():
    """"Google Cloud Platform (GCP) (...)" and "GCP (...)" are one skill, printed once."""
    from backend.analyzer.resume_schema import _same_skill, canonicalize_tech
    assert _same_skill("Google Cloud Platform (GCP) (Cloud Run, GKE)", "GCP (Cloud Run, GKE)")
    assert _same_skill("Retrieval-Augmented Generation (RAG)", "RAG")
    assert _same_skill("Amazon Web Services (AWS)", "AWS")
    assert not _same_skill("AWS (ECS, Lambda)", "Azure (App Services)")

    # What the résumé prints is never rewritten by that rule.
    printed = "Google Cloud Platform (GCP) (Cloud Run, GKE, Firestore)"
    assert canonicalize_tech(printed) == printed

    base = {**BASE, "skills": {"Cloud": "Google Cloud Platform (GCP) (Cloud Run, GKE), AWS"}}
    reply = {**REPLY, "technical_skills": [{"category": "Cloud", "skills": "GCP (Cloud Run, GKE)"}]}
    line = build_tailored_resume(reply, base, "")["skills"]["Cloud"]
    assert line.lower().count("cloud run") == 1, line


def test_a_parenthesised_group_is_one_skill_not_many():
    """"AWS (ECS, Lambda)" split on every comma left "AWS (ECS" and a stray "Lambda)"."""
    from backend.analyzer.resume_schema import _split_skills
    line = "AWS (ECS, Lambda, S3), Microsoft Azure (App Services, Azure Functions), Docker"
    assert _split_skills(line) == ["AWS (ECS, Lambda, S3)",
                                   "Microsoft Azure (App Services, Azure Functions)",
                                   "Docker"]
    assert _split_skills("Docker, Kubernetes, Helm") == ["Docker", "Kubernetes", "Helm"]


def test_a_renamed_category_takes_its_leftovers_with_it():
    """The reply may merge categories; restoring the old heading printed orphan lines."""
    base = {**BASE, "skills": {
        "Cloud": "AWS (ECS, Lambda), Microsoft Azure (App Services), Google Cloud Platform (GCP)",
        "Containers & DevOps": "Docker, Kubernetes, IaC",
        "Tools & Methodologies": "Git, Jira, Agile"}}
    reply = {**REPLY, "technical_skills": [
        {"category": "Cloud & DevOps", "skills": "AWS (ECS, Lambda), Docker, Kubernetes"}]}
    out = build_tailored_resume(reply, base, "")["skills"]

    merged = out["Cloud & DevOps"]
    assert "Microsoft Azure (App Services)" in merged and "IaC" in merged
    assert merged.count("(") == merged.count(")")          # brackets never torn
    assert "Cloud" not in out and "Containers & DevOps" not in out   # no orphan headings
    # A heading that shares nothing keeps its own line rather than being swept in.
    assert out["Tools & Methodologies"] == "Git, Jira, Agile"


def test_a_skill_whose_name_sits_inside_another_is_not_swallowed():
    """"SQL" is inside "PostgreSQL": containment matching lost the same four terms every run."""
    from backend.analyzer.resume_schema import _same_skill
    for a, b in (("PostgreSQL", "SQL"), ("MySQL", "SQL"), ("SQL Server", "SQL"),
                 ("GitHub", "Git"), ("GitHub Actions", "GitHub"), ("Spring Data JPA", "JPA")):
        assert not _same_skill(a, b), f"{a} wrongly treated as {b}"

    # Genuinely the same product, by alias or by vendor prefix.
    for a, b in (("PostgreSQL", "Postgres"), ("PostgreSQL", "Postgres SQL"),
                 ("AWS CloudWatch", "CloudWatch"), ("Apache Cassandra", "Cassandra"),
                 ("Kubernetes", "K8s"), ("VLAN", "VLANs")):
        assert _same_skill(a, b), f"{a} should match {b}"


def test_the_databases_line_survives_a_reply_that_trims_it():
    """The case four job descriptions in a row lost: SQL present, PostgreSQL deleted."""
    base = {**BASE, "skills": {
        "Programming Languages": "Java, JavaScript, SQL, Python",
        "Databases": "PostgreSQL, Oracle, MySQL, SQL Server, MongoDB, Redis"}}
    reply = {**REPLY, "technical_skills": [
        {"category": "Programming Languages", "skills": "Java, JavaScript, SQL, Python"},
        {"category": "Databases", "skills": "Oracle, MongoDB, Redis"}]}
    kept = build_tailored_resume(reply, base)["skills"]["Databases"]
    for term in ("PostgreSQL", "MySQL", "SQL Server"):
        assert term in kept, f"{term} lost from {kept!r}"


def test_a_one_letter_language_survives_both_the_filter_and_the_backfill():
    """"C" is a substring of half a résumé, so containment would swallow it either way."""
    base = {**BASE, "skills": {"Languages": "Python, SQL, C"}}
    reply = {**REPLY, "technical_skills": [{"category": "Languages", "skills": "Python"}]}
    kept = build_tailored_resume(reply, base)["skills"]["Languages"]
    assert [s.strip() for s in kept.split(",")] == ["Python", "SQL", "C"]

    # And the reply may state it itself without being filtered out as ungrounded.
    reply_c = {**REPLY, "technical_skills": [{"category": "Languages", "skills": "C, Python"}]}
    assert "C" in [s.strip() for s in build_tailored_resume(reply_c, base)["skills"]["Languages"].split(",")]


def test_backfill_does_not_duplicate_a_skill_the_reply_reworded():
    """"VLAN" must not come back beside the reply's "VLANs"."""
    base = {**BASE, "skills": {"Networking": "VLAN, DHCP"}}
    reply = {**REPLY, "technical_skills": [{"category": "Networking", "skills": "VLANs"}]}
    assert build_tailored_resume(reply, base)["skills"]["Networking"] == "VLANs, DHCP"


# ── the worker ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_worker_sends_the_kb_sections_and_asks_for_the_schema(test_db, monkeypatch):
    base = Resume(id=uuid.uuid4(), name="Base", is_base=True, template="inter", json_data=BASE)
    job = Job(id=uuid.uuid4(), title="Senior Data Engineer", company="Acme Health",
              description="Azure Data Factory, Databricks, HIPAA", external_id="ext-ats-worker")
    test_db.add_all([base, job])
    test_db.add(Setting(key="cv_tailor_prompt", value="Resume:\n{resume_json}\n\nJD:\n{job_description}"))
    test_db.add(Setting(key="tailor_auto_quick_score", value="off"))
    test_db.commit()

    calls = []

    async def fake_call(prompt, system, max_tokens, **kwargs):
        calls.append({"prompt": prompt, "system": system, "max_tokens": max_tokens, **kwargs})
        return {"text": json.dumps(REPLY), "usage": {}}

    monkeypatch.setattr("backend.analyzer.llm_client.call_cv_tailor_llm", fake_call)
    import backend.api.routes_resumes as rr

    await rr._tailor_impl(str(base.id), str(job.id), None)

    (call,) = calls
    assert call["response_schema"] is TAILOR_JSON_SCHEMA
    assert call["schema_name"] == TAILOR_SCHEMA_NAME
    assert call["max_tokens"] == rr.TAILOR_MAX_TOKENS
    sent = json.loads(call["prompt"].split("Resume:\n", 1)[1].split("\n\nJD:", 1)[0])
    assert list(sent) == ["header", "summary", "skills", "experience", "projects", "education", "certifications"]
    assert "Truthfulness overrides every other instruction" in call["system"]

    test_db.expire_all()
    copy = test_db.query(Resume).filter(Resume.is_base == False).one()
    assert copy.json_data[LAYOUT_KEY] == ATS_LAYOUT
    assert copy.json_data["summary"] == REPLY["professional_summary"]


# ── the ladder ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tailor_call_walks_the_openrouter_ladder_with_the_schema(test_db, monkeypatch):
    import backend.analyzer.llm_client as L

    for key, value in {"cv_tailor_llm_provider": "openrouter", "cv_tailor_llm_model": "x/model",
                       "cv_tailor_llm_api_key": "k"}.items():
        test_db.add(Setting(key=key, value=value))
    test_db.commit()

    seen = []

    async def fake_dispatch(provider, model, api_key, prompt, system, max_tokens, **kwargs):
        seen.append((kwargs.get("backend"), kwargs.get("response_schema")))
        if len(seen) == 1:
            raise L.BackendUnavailableError("saturated")
        return {"text": "{}", "usage": {}}

    monkeypatch.setattr(L, "_dispatch", fake_dispatch)
    resp = await L.call_cv_tailor_llm("p", "s", max_tokens=100, response_schema=TAILOR_JSON_SCHEMA,
                                      schema_name=TAILOR_SCHEMA_NAME)

    assert [b for b, _ in seen] == [L.OPENROUTER_BACKEND_LADDER[0], L.OPENROUTER_BACKEND_LADDER[1]]
    assert all(schema is TAILOR_JSON_SCHEMA for _, schema in seen)
    assert (resp["provider"], resp["model"]) == ("openrouter", "x/model")


# ── the templates ───────────────────────────────────────────────────────────

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "resume_templates"


def _template_names():
    return sorted(p.name for p in TEMPLATES_DIR.iterdir() if (p / "template.html.j2").exists())


@pytest.mark.parametrize("name", _template_names())
def test_every_template_holds_hyphenated_compounds_together(name):
    """A line that ends in "processed-" is read back as "processedevent" by any reflowing parser."""
    from backend.api.routes_resumes import _render_html
    base = {**BASE, "summary": "Built a processed-event pipeline.",
            "skills": {"ETL / ELT": "Azure Data Factory, dual-currency reporting"},
            "experience": [{**BASE["experience"][0],
                            "bullets": ["Deduplicated against a persisted processed-event set."]}]}
    html = _render_html(base, name, "letter")
    body = html.split("<body>", 1)[1]
    assert ".nb { white-space: nowrap; }" in html
    # in a bullet (via the bold filter) and in the summary and skills (via nb)
    assert body.count('<span class="nb">processed-event</span>') >= 2
    assert '<span class="nb">dual-currency</span>' in body
    # the hyphen stays a plain ASCII one, so an ATS keyword search still matches
    assert "\u2011" not in html


@pytest.mark.parametrize("name", _template_names())
def test_every_template_prints_a_tailored_copy_in_kb_order(name):
    from backend.api.routes_resumes import _render_html
    html = _render_html(build_tailored_resume(REPLY, BASE), name, "letter")
    text = " ".join(re.sub(r"<[^>]+>", " ", html.split("<body>", 1)[1]).split())
    marks = ["Shah Ahsan Ali", "Professional Summary", "Technical Skills", "Work Experience",
             "Projects", "Education", "Certifications"]
    at = [text.find(m) for m in marks]
    assert -1 not in at and at == sorted(at), dict(zip(marks, at))
    assert "A paper" not in text


# --- a bullet may not name a tool its own role never named (rule 0 as a check) ----------

def _devops_base():
    return {
        "header": {"name": "A"}, "summary": "", "skills": {
            "Cloud Platforms": "AWS (EC2, EKS, S3), GCP (GKE, Compute Engine), Microsoft Azure (AKS)",
            "Containers & Orchestration": "Docker, Kubernetes, EKS, AKS, GKE, Helm, ArgoCD",
            "IaC": "Terraform, Ansible"},
        "experience": [{"company": "C", "title": "SRE", "bullets": [
            "Implemented multi-cloud platform standards across AWS, Azure, and GCP by aligning "
            "Kubernetes operations and Terraform modules.",
            "Implemented automated remediation workflows for recurring Kubernetes issues.",
            "Built and maintained clusters on Amazon EKS for production microservices."]}],
        "projects": [], "education": [], "certifications": []}


def _reply_with(bullets):
    return {"professional_summary": "", "technical_skills": [], "projects": [],
            "education": [], "certifications": [],
            "header": {"name": "", "title": "", "location": "", "email": "", "phone": "",
                       "work_authorization": "", "github": "", "linkedin": ""},
            "work_experience": [{"company": "C", "job_title": "SRE", "bullets": bullets}]}


def test_a_bullet_that_narrows_kubernetes_to_gke_keeps_the_resumes_own_sentence():
    """Measured: on a GCP posting the model rewrote "Kubernetes operations" to "GKE operations"
    in every run. The résumé's Kubernetes work is EKS, so that narrowing is a claim it never made."""
    base = _devops_base()
    out = build_tailored_resume(_reply_with([
        "Implemented multi-cloud platform standards across GCP, AWS, and Azure by aligning "
        "GKE operations and Terraform modules.",
        "Implemented automated remediation workflows for recurring GKE issues.",
        "Built and maintained clusters on Amazon EKS for production microservices."]), base)
    kept = out["experience"][0]["bullets"]
    assert kept[0] == base["experience"][0]["bullets"][0]
    assert kept[1] == base["experience"][0]["bullets"][1]
    assert not any("GKE" in b for b in kept[:2])


def test_a_genuine_rewrite_that_adds_no_new_tool_is_left_alone():
    base = _devops_base()
    reworded = ("Aligned Kubernetes operations and Terraform modules to set multi-cloud platform "
                "standards across AWS, Azure, and GCP.")
    out = build_tailored_resume(_reply_with(
        [reworded] + base["experience"][0]["bullets"][1:]), base)
    assert out["experience"][0]["bullets"][0] == reworded


def test_a_tool_the_role_uses_in_another_bullet_may_move_between_them():
    """EKS is named in the role's third bullet, so mentioning it in the first is not a new claim."""
    base = _devops_base()
    moved = "Implemented multi-cloud standards across AWS, Azure, and GCP, aligning EKS operations."
    out = build_tailored_resume(_reply_with(
        [moved] + base["experience"][0]["bullets"][1:]), base)
    assert out["experience"][0]["bullets"][0] == moved


def test_the_vocabulary_comes_from_the_resume_so_another_stack_is_covered_too():
    """Nothing about technology is hardcoded: a computer-vision résumé polices its own terms."""
    base = {"header": {"name": "A"}, "summary": "", "skills": {
        "CV Frameworks": "PyTorch, Ultralytics (YOLO), MediaPipe, PaddleOCR, OpenCV",
        "Web": "FastAPI, Streamlit"},
        "experience": [{"company": "C", "title": "CV Engineer", "bullets": [
            "Trained detection models in PyTorch and served them behind FastAPI.",
            "Built an OCR pipeline for scanned documents."]}],
        "projects": [], "education": [], "certifications": []}
    out = build_tailored_resume(_reply_with([
        "Trained YOLO detection models in PyTorch and served them behind FastAPI.",
        "Built a PaddleOCR pipeline for scanned documents."]), base)
    kept = out["experience"][0]["bullets"]
    assert kept == base["experience"][0]["bullets"], "neither YOLO nor PaddleOCR was in the role"


def test_a_drifted_bullet_with_no_source_to_fall_back_on_is_kept_not_dropped():
    """Losing real content costs more than carrying one flagged bullet."""
    base = _devops_base()
    invented = "Migrated the data warehouse to GKE using a bespoke controller."
    out = build_tailored_resume(_reply_with(
        base["experience"][0]["bullets"] + [invented]), base)
    assert invented in out["experience"][0]["bullets"]
    assert len(out["experience"][0]["bullets"]) == 4


def test_preview_renders_the_template_asked_for_without_touching_the_stored_one():
    """The Streamlit app always shows Word Classic; it must not PATCH the résumé to do it."""
    from backend.api.routes_resumes import _render_html
    word = _render_html(BASE, "word", "letter")
    inter = _render_html(BASE, "inter", "letter")
    assert "Carlito" in word and "Carlito" not in inter


def test_both_llm_calls_state_how_much_thinking_they_want():
    """Neither call may leave the effort level unset.

    Omitting it takes the CLI's own default, which is the most expensive one: measured,
    that cost 69s and 6,445 thinking tokens where the same call at low effort took 16s
    and 477. The levels themselves are a tuning choice; stating one is not.

    Measured on Claude Sonnet: import at low effort structures a 76-bullet résumé in 9s
    with nothing lost (it was 73-125s before), while tailoring at high rewrites a third of
    the bullets on every run where medium's rewriting swung between 1 and 6 of 24.
    """
    import inspect
    from backend.api import routes_resumes as rr

    tailor = inspect.getsource(rr._tailor_impl)
    assert "reasoning=" in tailor, "tailoring must state an effort level, not leave it to the default"

    parse = inspect.getsource(rr._parse_llm)
    assert "reasoning=False" in parse, "import must say it is transcription"


def test_claude_code_turns_that_into_an_effort_level():
    """`reasoning` has to reach the CLI as --effort, or the settings above are decoration.

    Which level each value maps to is a tuning choice and moves with experience, so this
    pins the mechanism rather than the mapping: an --effort flag is always sent, and its
    value comes from `reasoning` rather than being left to the CLI's own default — which
    is the expensive one (measured: 69s and 6,445 thinking tokens, against 16s and 477
    at low).
    """
    import inspect
    from backend.analyzer.llm_client import _call_claude_code
    src = inspect.getsource(_call_claude_code)
    assert '"--effort"' in src, "the flag must always be sent"
    assert "reasoning" in src.split('"--effort"')[1].split("\n")[0], \
        "the level must be derived from `reasoning`, not hardcoded"
