"""The posting's search terms, matched the way a recruiter's ATS search matches them. No model involved."""
import pytest

from backend.analyzer import posting_keywords as keywords

RESUME = """Priya Raman
priya@example.com | +1 415 555 0134
## Summary
Backend engineer building RESTful APIs.
## Experience
Senior Java Developer at Acme (Jan 2020 – Present — Austin, TX
- Ran K8s clusters and a CI CD pipeline for NodeJS services on Postgres
- Wrote integration tests
## Skills
Languages: Java, Python
"""


@pytest.mark.parametrize("term", ["REST APIs", "Kubernetes", "CI/CD", "Node.js", "PostgreSQL", "integration test",
                                  "java"])
def test_the_safe_differences_still_match(term):
    """Plurals, -ful, spacing and punctuation, and the dictionary's other spellings of the same tool."""
    assert keywords.appears_in(term, RESUME), term


@pytest.mark.parametrize("term", ["SQL", "Amazon EKS", "Go", "GraphQL"])
def test_a_narrower_name_or_an_absent_term_does_not_match(term):
    """PostgreSQL is not SQL, K8s is not EKS, and "Go" is a word, not the language, unless it is written as one."""
    assert not keywords.appears_in(term, RESUME), term


def test_a_term_its_posting_line_does_not_contain_is_dropped():
    kept, dropped = keywords.grounded(["Kafka", "Spring Boot", "event streaming"], "Strong Kafka and Spring Boot")
    assert kept == ["Kafka", "Spring Boot"] and dropped == ["event streaming"]


def test_the_same_term_asked_twice_counts_once_at_its_most_important_kind():
    part = keywords.keyword_part([{"kind": "nice", "terms": ["K8s"], "wording": "exact"},
                                  {"kind": "must", "terms": ["Kubernetes"], "wording": "exact"},
                                  {"kind": "must", "terms": ["Terraform"], "wording": "none"}], RESUME)
    assert [row["term"] for row in part["terms"]] == ["Kubernetes", "Terraform"]
    assert part["found"] == ["Kubernetes"] and part["missing"] == ["Terraform"]
    assert part["share"] == 0.5


def test_another_name_that_was_checked_earns_half_and_kinds_weigh_three_two_one():
    part = keywords.keyword_part([{"kind": "must", "terms": ["Python"], "wording": "exact"},
                                  {"kind": "duty", "terms": ["Terraform"], "wording": "other"},
                                  {"kind": "nice", "terms": ["Rust"], "wording": "none"}], RESUME)
    assert part["other_name"] == ["Terraform"]
    assert part["share"] == round((3 * 1.0 + 2 * 0.5 + 1 * 0.0) / 6, 3)


def test_where_a_found_term_is_written():
    part = keywords.keyword_part([{"kind": "must", "terms": ["Python", "Kubernetes"], "wording": "exact"}], RESUME)
    where = {row["term"]: row["where"] for row in part["terms"]}
    assert where == {"Python": "listed", "Kubernetes": "in a role"}


def test_wording_is_exact_other_none_or_not_applicable():
    assert keywords.wording_of(["Kafka", "PostgreSQL"], RESUME, other_name_checked=False) == "exact"
    assert keywords.wording_of(["Terraform"], RESUME, other_name_checked=True) == "other"
    assert keywords.wording_of(["Terraform"], RESUME, other_name_checked=False) == "none"
    assert keywords.wording_of([], RESUME, other_name_checked=False) is None


def test_no_terms_leaves_the_part_out():
    assert keywords.keyword_part([{"kind": "duty", "terms": [], "wording": None}], RESUME)["share"] is None
