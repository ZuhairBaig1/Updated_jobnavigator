"""The posting's skills a résumé names and misses come from a dictionary lookup, the same way every time."""
import pytest

from backend.analyzer.keyword_match import keyword_match, skills_in

POSTING = ("We need an engineer with strong Python and FastAPI skills. You will go through our pipelines in "
           "Azure Data Factory and deploy with Docker and Kubernetes.")
RESUME = "Built data pipelines in Python and Azure Data Factory. Deployed services with docker. Set up CI/CD."


def test_the_postings_skills_are_split_into_matched_and_missing():
    result = keyword_match(RESUME, POSTING)

    assert result["matched_keywords"] == ["Python", "Azure Data Factory", "Docker"]
    assert result["missing_keywords"] == ["FastAPI", "Kubernetes"]
    assert result["keyword_coverage_pct"] == 60


def test_a_skill_the_posting_does_not_ask_for_is_left_out():
    result = keyword_match(RESUME, POSTING)

    assert "CI/CD" not in result["matched_keywords"] + result["missing_keywords"]


def test_the_longest_phrase_is_one_skill():
    assert "Azure Data Factory" in skills_in(POSTING)
    assert "Microsoft Azure" not in skills_in(POSTING)


@pytest.mark.parametrize("text, skill", [
    ("Deployed with Docker", "Docker"),
    ("deployed with docker", "Docker"),
    ("Built agents in LangGraph", "LangGraph"),
    ("built agents in langgraph", "LangGraph"),
    ("Scored resumes with Jev", "Jev"),
    ("Packaged every service as a container", "Containers"),
    ("Led the containerization of legacy apps", "Containers"),
    ("Built llm apps", "Large Language Models"),
    ("Shipped LLMs to production", "Large Language Models"),
    ("Integrated the OpenAI API", "OpenAI"),
    ("Used Open AI models", "OpenAI"),
    ("Automated reviews with Claude Code", "Claude Code"),
    ("Deployed to k8s", "Kubernetes"),
    ("Wrote services in nodejs", "Node.js"),
    ("Tuned GPT-4o prompts", "GPT"),
])
def test_a_skill_is_found_however_it_is_capitalised(text, skill):
    assert skill in skills_in(text)


@pytest.mark.parametrize("text, skill, is_found", [
    ("Wrote services in Go and Swift", "Golang", True),
    ("You will go through our swift review process", "Golang", False),
    ("Built a RAG assistant", "Retrieval-Augmented Generation", True),
    ("Cleaned it with a rag", "Retrieval-Augmented Generation", False),
    ("Used Cursor daily", "Cursor", True),
    ("Moved the cursor to the next row", "Cursor", False),
])
def test_a_tool_name_that_is_also_an_ordinary_word_needs_its_capital(text, skill, is_found):
    assert (skill in skills_in(text)) is is_found


def test_ordinary_words_are_not_skills():
    found = skills_in("The role is based in one location and reports to the product team every time.")

    assert found == {}


def test_a_posting_with_no_known_skills_has_no_coverage_figure():
    result = keyword_match(RESUME, "We are looking for a kind and curious person.")

    assert result == {"matched_keywords": [], "missing_keywords": [], "keyword_coverage_pct": None}


def test_how_often_a_skill_is_mentioned_is_counted():
    assert skills_in("Python here, python there, and PYTHON again")["Python"]["count"] == 3


GENERAL_POSTING = ("Strong communication and leadership. Own monitoring, automation and scalability. "
                   "Agile team doing DevOps and machine learning.")


def test_general_abilities_are_never_listed_as_missing_tools():
    result = keyword_match("Wrote Python.", GENERAL_POSTING)

    assert result == {"matched_keywords": [], "missing_keywords": [], "keyword_coverage_pct": None}


def test_general_abilities_are_recognised_as_soft_skills_or_other_keywords():
    found = skills_in(GENERAL_POSTING)

    assert found["Communication"]["type"] == "soft"
    assert found["Monitoring"]["type"] == "other"
    assert all(mention["type"] != "hard" for mention in found.values())


def test_named_techniques_a_posting_asks_for_are_kept():
    found = skills_in("Build RAG pipelines and CI/CD for our microservices, with infrastructure as code.")

    assert set(found) == {"Retrieval-Augmented Generation", "CI/CD", "Microservices", "Infrastructure as Code"}
