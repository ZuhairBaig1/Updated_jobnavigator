"""Job requirement extraction keeps only what the posting states, one requirement per item."""
import pytest

from backend.analyzer import job_requirements as jr

POSTING = (
    "Senior DevOps Engineer. Must have: GKE, Terraform, Jenkins. Strong understanding of security, "
    "IAM, and secrets management. Good to have: Datadog. 5+ years required. Key Skills "
    "JenkinsGitHub/GitLabCI/CD pipelines using tools such as Cloud BuildTerraform GCP Cloud computing"
)


def _item(name, quote, aliases=()):
    return {"name": name, "aliases": list(aliases), "quote": quote}


def test_clean_up_splits_glued_skill_names_without_breaking_real_ones():
    cleaned = jr.clean_job_text("Key Skills JenkinsGitHub/GitLabCI/CD pipelines using Cloud BuildTerraform GCP")
    assert "Jenkins GitHub/GitLab CI/CD" in cleaned
    assert "Cloud Build Terraform GCP" in cleaned
    assert "GitHub" in cleaned  # a known name is never split itself


def test_clean_up_removes_markdown_leftovers():
    cleaned = jr.clean_job_text("**Experience:** 7\\+ Years\n-----------\n### Skills\nAI \\& ML")
    assert cleaned == "Experience: 7+ Years\n\nSkills\nAI & ML"


def test_items_without_proof_in_the_posting_are_dropped():
    raw = {"must_have": [_item("Terraform", "Terraform"), _item("Kafka", "Apache Kafka streaming")]}
    result = jr._check_and_tidy(raw, jr.clean_job_text(POSTING))
    assert [item["name"] for item in result["must_have"]] == ["Terraform"]
    assert result["dropped_unverified"] == ["Kafka"]


def test_an_item_with_a_stitched_quote_is_kept_when_its_name_is_in_the_posting():
    raw = {"must_have": [_item("Terraform", "Must have hands-on experience with Terraform"),
                         _item("Google Kubernetes Engine", "hands-on experience with Kubernetes on Google", ["GKE"]),
                         _item("Kafka", "Must have hands-on experience with Kafka", ["Apache Kafka"])]}
    result = jr._check_and_tidy(raw, jr.clean_job_text(POSTING))
    assert [item["name"] for item in result["must_have"]] == ["Terraform", "Google Kubernetes Engine"]
    assert result["dropped_unverified"] == ["Kafka"]


def test_bundled_items_are_split_and_duplicates_keep_the_must_have():
    raw = {
        "must_have": [_item("Security, IAM, and secrets management",
                            "security, IAM, and secrets management")],
        "nice_to_have": [_item("IAM", "IAM"), _item("Datadog", "Datadog")],
    }
    result = jr._check_and_tidy(raw, jr.clean_job_text(POSTING))
    assert [item["name"] for item in result["must_have"]] == ["Security", "IAM", "secrets management"]
    assert [item["name"] for item in result["nice_to_have"]] == ["Datadog"]


@pytest.mark.parametrize("years, expected", [(5, 5), ("5", 5), (7, None), (0, None), (99, None), (None, None), (True, None)])
def test_years_must_be_sensible_and_stated_in_the_posting(years, expected):
    assert jr._check_and_tidy({"min_years": years}, POSTING)["min_years"] == expected


@pytest.mark.parametrize("posting, claimed, expected", [
    ("On-site Opportunity-No. GKE and Terraform.", "remote", ""),
    ("This is a fully remote role.", "remote", "remote"),
    ("Hybrid, 3 days per week in office.", "hybrid", "hybrid"),
    ("Work Location: In person", "onsite", "onsite"),
    ("Location: Hyderabad, India", "onsite", ""),
])
def test_work_mode_is_kept_only_when_the_posting_says_so(posting, claimed, expected):
    assert jr._check_and_tidy({"work_mode": claimed}, posting)["work_mode"] == expected


def test_text_fields_the_posting_does_not_contain_are_left_empty_and_reported():
    raw = {"title": "Senior DevOps Engineer", "education": "Master's degree in Physics",
           "location": "Berlin", "seniority": "Senior", "work_mode": "remote", "min_years": 12}
    result = jr._check_and_tidy(raw, POSTING)
    assert result["title"] == "Senior DevOps Engineer"
    assert result["seniority"] == "Senior"
    assert result["education"] == ""
    assert result["location"] == ""
    assert sorted(result["unverified_fields"]) == ["education", "location", "min_years", "work_mode"]


def test_a_combined_title_keeps_the_parts_the_posting_contains():
    raw = {"title": "Platform Lead / Senior DevOps Engineer"}
    assert jr._check_and_tidy(raw, POSTING)["title"] == "Senior DevOps Engineer"


def test_a_reworded_location_keeps_each_place_the_posting_names():
    posting = "Location: Any Location. Preferred Locations: Chennai, Hyderabad, Pune"
    raw = {"location": "Any Location (Preferred: Chennai, Hyderabad, Pune)"}
    assert jr._check_and_tidy(raw, posting)["location"] == "Any Location, Preferred, Chennai, Hyderabad, Pune"


def test_known_skills_the_model_missed_are_flagged_not_added():
    raw = {"must_have": [_item("Terraform", "Terraform"), _item("Google Kubernetes Engine", "GKE", ["GKE"])]}
    result = jr._check_and_tidy(raw, jr.clean_job_text(POSTING))
    assert "Jenkins" in result["also_mentioned"]
    assert "Google Kubernetes Engine" not in result["also_mentioned"]
    assert [item["name"] for item in result["must_have"]] == ["Terraform", "Google Kubernetes Engine"]


@pytest.mark.asyncio
async def test_schema_method_asks_for_the_fixed_form(monkeypatch):
    calls = []

    async def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return {"text": '{"title": "Senior DevOps Engineer", "min_years": 5, '
                        '"must_have": [{"name": "Terraform", "aliases": [], "quote": "Terraform"}], '
                        '"nice_to_have": [], "certifications": [], "responsibilities": []}'}

    monkeypatch.setattr(jr, "call_llm", fake_call_llm)
    result = await jr.extract_job_requirements(POSTING, "claude_code", "claude-haiku-4-5")

    assert calls[0]["response_schema"] is jr.REQUIREMENTS_SCHEMA
    assert calls[0]["reasoning"] is False
    assert "JOB POSTING" in calls[0]["prompt"]
    assert result["title"] == "Senior DevOps Engineer"
    assert result["min_years"] == 5
    assert result["method"] == "schema"


def test_langextract_results_without_a_location_in_the_posting_are_ignored():
    class Extraction:
        def __init__(self, kind, text, attributes=None, located=True):
            self.extraction_class = kind
            self.extraction_text = text
            self.attributes = attributes
            self.char_interval = object() if located else None

    fields = jr._langextract_to_fields([
        Extraction("requirement", "GKE", {"priority": "must", "name": "Google Kubernetes Engine"}),
        Extraction("requirement", "Datadog", {"priority": "nice", "name": "Datadog"}),
        Extraction("requirement", "Kafka", {"priority": "must"}, located=False),
        Extraction("min_years", "5+ years", {"years": "5"}),
        Extraction("work_mode", "Remote", {"mode": "Remote"}),
    ])
    assert [item["name"] for item in fields["must_have"]] == ["Google Kubernetes Engine"]
    assert [item["quote"] for item in fields["nice_to_have"]] == ["Datadog"]
    assert fields["min_years"] == 5
    assert fields["work_mode"] == "remote"
