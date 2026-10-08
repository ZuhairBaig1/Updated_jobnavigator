"""A posting is split into sections by patterns alone, so the same text always splits the same way."""
from backend.analyzer.job_sections import split_posting

POSTING = """Senior Data Engineer
Location: As per requirement
Experience: 6+ Years

What You'll Do
- Build pipelines in Azure Data Factory
- Lead design reviews

What You Have
- 6+ years building data platforms, including 3 years with Spark
- Python SQL Kubernetes
- B.Tech in Computer Science
- AWS Certified Solutions Architect is preferred
- Must be authorized to work in the US

Bonus Points
- Kafka, Airflow

Perks and Benefits
- Health insurance
"""


def test_lines_land_under_the_section_their_heading_opens():
    sections = split_posting(POSTING)

    assert "Build pipelines in Azure Data Factory" in sections["responsibilities"]
    assert "Python SQL Kubernetes" in sections["must_have"]
    assert "Kafka, Airflow" in sections["nice_to_have"]
    assert sections["ignored"] == ["Health insurance"]


def test_a_labelled_value_is_not_read_as_a_heading():
    sections = split_posting(POSTING)

    assert "Location: As per requirement" in sections["posting_details"]
    assert "Location: As per requirement" not in sections["must_have"]


def test_a_line_that_calls_itself_preferred_is_nice_to_have_under_any_heading():
    sections = split_posting(POSTING)

    assert "AWS Certified Solutions Architect is preferred" in sections["nice_to_have"]
    assert "AWS Certified Solutions Architect is preferred" not in sections["must_have"]


def test_the_facts_patterns_can_read():
    sections = split_posting(POSTING)

    assert sections["years"] == "6+"
    assert sections["education"] == ["B.Tech in Computer Science"]
    assert sections["work_authorization"] == ["Must be authorized to work in the US"]
    assert sections["certifications"] == ["AWS Certified Solutions Architect is preferred"]


def test_skills_are_found_whatever_separates_them():
    sections = split_posting(POSTING)

    assert {"Python", "SQL", "Kubernetes", "Apache Spark"} <= set(sections["must_have_skills"])
    assert "Apache Kafka" in sections["nice_to_have_skills"]


UNHEADED_POSTING = """We are a payments company growing fast across Europe.
You will build the pipelines that feed our fraud models.
Strong experience with Python and SQL in production.
Reference Architectures: Contribute to and maintain reusable patterns.
"""


def test_a_line_under_no_heading_is_filed_by_how_it_reads():
    sections = split_posting(UNHEADED_POSTING)

    assert sections["general"] == ["We are a payments company growing fast across Europe."]
    assert sections["must_have"] == ["Strong experience with Python and SQL in production."]
    assert sections["responsibilities"] == [
        "You will build the pipelines that feed our fraud models.",
        "Reference Architectures: Contribute to and maintain reusable patterns.",
    ]


PARAGRAPH_POSTING = (
    "Job description\n"
    "Generative AI Architect. Strong expertise in GenAI frameworks such as LangChain and Azure OpenAI and in "
    "the wider set of AI platforms used by enterprise clients today. Key Responsibilities Design and deliver "
    "agentic AI architectures for enterprise clients. Mentor engineers and contribute to practice accelerators.\n"
)


def test_a_heading_inside_a_paragraph_still_opens_its_section():
    sections = split_posting(PARAGRAPH_POSTING)

    assert sections["responsibilities"] == [
        "Design and deliver agentic AI architectures for enterprise clients.",
        "Mentor engineers and contribute to practice accelerators.",
    ]
    assert any("LangChain" in line for line in sections["must_have"])


CLUTTERED_POSTING = """Key Responsibilities
Monitoring & Reliability
- Build dashboards in Grafana
- Role: DevOps Engineer
- Work Timings 2:00PM to 11:00 PM IST

Requirements
- Five years with Kubernetes
- We are proud of our inclusive culture.
- We are an equal opportunity employer.
- #LI-Onsite
"""


def test_lines_that_are_not_content_are_kept_out_of_the_sections():
    sections = split_posting(CLUTTERED_POSTING)

    assert sections["responsibilities"] == ["Build dashboards in Grafana"]
    assert sections["must_have"] == ["Five years with Kubernetes"]
    assert "We are proud of our inclusive culture." in sections["about"]
    assert "We are an equal opportunity employer." in sections["ignored"]


import pytest


@pytest.mark.parametrize("line, years", [
    ("1-5 years of relevant software engineering experience", "1-5"),
    ("5+ years of experience in DevOps", "5+"),
    ("2 to 3 yrs working with data", "2-3"),
    ("At least 4 years of backend work", "4"),
    ("8 plus years building platforms", "8+"),
])
def test_years_are_kept_the_way_the_posting_states_them(line, years):
    assert split_posting("Requirements\n- " + line)["years"] == years


JOB_BOARD_POSTING = """Job description

Preferred candidate profile

Primary Skill : Devops , Azure, IAAS, Ci/CD

Secondary: Terraform,Ansible

Exp: 6 to 12

Location : India /Remote

Notice Period : Immediate to 45 Days

Role & responsibilities :

    Strong experience in cloud infrastructure (Azure preferred) and distributed systems
    Proficiency in programming/scripting languages such as Python, Java
    Proven track record of delivering and leading large-scale infrastructure initiatives
    Design and run the CI/CD pipelines for the platform team

Role: Software Development - Other
Education
UG: Any Graduate
Key Skills
Skills highlighted with '' are preferred keyskills
DevOpsIaaSazure
Ci/Cd
"""


def test_a_job_board_posting_with_labelled_skill_lists():
    sections = split_posting(JOB_BOARD_POSTING)

    assert "Primary Skill : Devops , Azure, IAAS, Ci/CD" in sections["must_have"]
    assert sections["nice_to_have"] == ["Secondary: Terraform,Ansible"]
    assert sections["years"] == "6-12"
    assert sections["education"] == ["UG: Any Graduate"]
    assert "Ci/Cd" in sections["must_have"]
    assert "Notice Period : Immediate to 45 Days" in sections["posting_details"]


def test_a_requirement_filed_under_a_duties_heading_is_still_a_requirement():
    sections = split_posting(JOB_BOARD_POSTING)

    assert sections["responsibilities"] == ["Design and run the CI/CD pipelines for the platform team"]
    assert "Proficiency in programming/scripting languages such as Python, Java" in sections["must_have"]
    assert "Strong experience in cloud infrastructure (Azure preferred) and distributed systems" in sections["must_have"]
    assert "Proven track record of delivering and leading large-scale infrastructure initiatives" in sections["must_have"]


TITLE_BAR_POSTING = """DevOps Senior Engineer | Hyderabad | 10-15 Years

Mandatory Skills
    Experience with MongoDB/NoSQL Databases

Preferred Qualifications
    AWS DevOps Certification

Domain: Technology / Product Engineering / Cloud & DevOps.
"""


def test_a_title_bar_and_a_domain_label_are_posting_details():
    sections = split_posting(TITLE_BAR_POSTING)

    assert sections["must_have"] == ["Experience with MongoDB/NoSQL Databases"]
    assert sections["nice_to_have"] == ["AWS DevOps Certification"]
    assert sections["years"] == "10-15"
    assert "SQL" not in sections["must_have_skills"]


LEAD_IN_POSTING = """Job description
Acme runs payment systems for half the banks in Europe and has done so for twenty years.
To ensure you are set up for success, you will bring the following skillset & experience:

    6+ years of experience building and operating SaaS products in production environments.
    Demonstrated experience handling production incidents and postmortems.
    Direct, accountable leadership style with comfort owning operational outcomes.

Whilst these are nice to have, our team can help you develop in the following skills:

    Prior experience with our software stack and internal platforms.
    Formal SRE or cloud certifications

CA-DNP
Our commitment to you!
Acme's culture is built around its people, and nobody here is known just by an employee number.
Key Skills
Skills highlighted with '' are preferred keyskills
Automated Testing
AutomationAzureTest AutomationAutomatingCloud CertificationsDebuggingSaasSreCloud Engineering
"""


def test_a_sentence_that_introduces_a_list_opens_its_section():
    sections = split_posting(LEAD_IN_POSTING)

    assert sections["must_have"][:3] == [
        "6+ years of experience building and operating SaaS products in production environments.",
        "Demonstrated experience handling production incidents and postmortems.",
        "Direct, accountable leadership style with comfort owning operational outcomes.",
    ]
    assert sections["nice_to_have"] == ["Prior experience with our software stack and internal platforms.",
                                        "Formal SRE or cloud certifications"]


def test_a_paragraph_after_a_list_ends_the_list():
    sections = split_posting(LEAD_IN_POSTING)
    filed = sections["must_have"] + sections["nice_to_have"]

    assert not any("culture is built around" in line for line in filed)
    assert "CA-DNP" in sections["posting_details"]


def test_long_job_board_tag_lines_are_dropped_and_short_ones_kept():
    sections = split_posting(LEAD_IN_POSTING)

    assert "Automated Testing" in sections["must_have"]
    assert not any("AutomationAutomatingCloud" in line for line in sections["must_have"] + sections["certifications"])
