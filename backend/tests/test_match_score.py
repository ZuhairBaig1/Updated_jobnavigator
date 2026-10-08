"""The keyword match score: hard skills 60, soft skills 15, other keywords 5, job title 10, degree 10."""
import pytest

from backend.analyzer.match_score import _credit, match_score

POSTING = ("Senior Data Engineer. Bachelor's degree required. Build pipelines in Python, with Kafka and Kubernetes. "
           "Python is used across the team. Strong communication. Agile delivery.")


@pytest.mark.parametrize("times_in_posting, times_in_resume, credit", [
    (1, 0, 0.0),
    (4, 1, 0.625),
    (4, 2, 0.75),
    (2, 2, 1.0),
    (1, 5, 1.0),
])
def test_a_keyword_earns_credit_by_how_often_it_is_mentioned(times_in_posting, times_in_resume, credit):
    assert _credit(times_in_posting, times_in_resume) == credit


def test_every_part_is_earned_by_a_resume_that_has_it_all():
    resume = ("Senior Data Engineer. Bachelor's degree in Computer Science. Built Python services; Python again. "
              "Ran Kafka on Kubernetes. Clear communication in an Agile team.")

    result = match_score(resume, POSTING, "Senior Data Engineer")

    assert result["match_score"] == 100
    assert result["match_breakdown"] == {"hard": 60.0, "soft": 15.0, "other": 5.0, "title": 10, "degree": 10}


def test_hard_skills_are_worth_sixty_points_shared_between_them():
    resume = "Senior Data Engineer. Bachelor's degree. Python and more Python. Clear communication. Agile."

    result = match_score(resume, POSTING, "Senior Data Engineer")

    assert result["match_breakdown"]["hard"] == 20.0
    # A third of the hard skills, so a third of every other part: 20 + 5 + 1.7 + 3.3 + 3.3.
    assert result["match_score"] == 33


def test_one_mention_of_a_keyword_the_posting_repeats_earns_most_of_it():
    table = match_score("Python once.", POSTING)["keyword_table"]
    python = next(row for row in table if row["keyword"] == "Python")

    assert (python["job"], python["resume"], python["credit"]) == (2, 1, 0.75)


def test_a_part_the_posting_lacks_is_left_out_and_never_scores_free_points():
    result = match_score("Wrote Java.", "We use Python.")

    assert result["match_breakdown"] == {"hard": 0.0}
    assert result["match_score"] == 0


def test_the_points_of_a_missing_part_go_to_the_others_by_their_weight():
    result = match_score("Wrote Python. B.Tech in Computer Science.", "We use Python. Bachelor's degree required.")

    # Hard skills weigh 60 and the degree 10, so of 100 points they hold 85.7 and 14.3.
    assert result["match_out_of"] == {"hard": 85.7, "degree": 14.3}
    assert result["match_score"] == 100


def test_all_five_parts_keep_their_usual_points_when_the_posting_has_them_all():
    result = match_score("Wrote Python.", POSTING, "Senior Data Engineer")

    assert result["match_out_of"] == {"hard": 60.0, "soft": 15.0, "other": 5.0, "title": 10.0, "degree": 10.0}


ALL_THE_HARD_SKILLS = "\nPython, Python, Kafka, Kubernetes"


@pytest.mark.parametrize("resume, points", [
    ("Worked as a Senior Data Engineer for five years", 10),
    ("Worked as a senior data engineer", 10),
    ("Data Engineer, Senior level", 10),
    ("Senior Engineer on the platform team\nBuilt data pipelines", 0),
    ("Worked as a Data Engineer", 0),
])
def test_the_job_title_counts_when_one_line_has_all_its_words_in_any_order(resume, points):
    result = match_score(resume + ALL_THE_HARD_SKILLS, POSTING, "Senior Data Engineer")

    assert result["match_breakdown"]["title"] == points


def test_every_other_part_counts_only_as_far_as_the_hard_skills_are_met():
    resume = "Senior Data Engineer. Bachelor's degree. Clear communication in an Agile team."

    result = match_score(resume, POSTING, "Senior Data Engineer")

    # The title, the degree, the soft skill and the other keyword are all there, and none of the hard skills.
    assert result["match_breakdown"] == {"hard": 0.0, "soft": 0.0, "other": 0.0, "title": 0.0, "degree": 0.0}
    assert result["match_details"]["shown_before_scaling"]["title"] == 1.0


def test_a_posting_without_a_title_has_no_title_part():
    assert "title" not in match_score("Wrote Python.", POSTING, "")["match_breakdown"]


@pytest.mark.parametrize("resume, posting, share_of_degree_points", [
    ("M.Tech in Computer Science", "Bachelor's degree required", 1.0),
    ("B.Tech in Computer Science", "Bachelor's degree required", 1.0),
    ("B.Tech in Computer Science", "Master's degree required", 0.5),
    ("Built pipelines for six years", "Bachelor's degree required", 0.0),
    ("B.Tech in Computer Science", "Bachelor's required, Master's preferred", 0.5),
])
def test_the_degree_is_compared_with_what_the_posting_asks(resume, posting, share_of_degree_points):
    result = match_score(resume, posting)

    assert result["match_breakdown"]["degree"] == share_of_degree_points * result["match_out_of"]["degree"]


def test_a_posting_that_asks_for_no_degree_has_no_degree_part():
    assert "degree" not in match_score("Wrote Python.", "We use Python.")["match_breakdown"]


def test_a_posting_with_nothing_to_compare_has_no_score():
    assert match_score("Built pipelines for six years", "We need a strong engineer")["match_score"] is None


def test_the_table_lists_missing_hard_skills_first():
    table = match_score("Wrote Python.", POSTING)["keyword_table"]

    assert [row["keyword"] for row in table[:2]] == ["Kafka", "Kubernetes"]
    assert table[0]["resume"] == 0


SOFT_POSTING = "We use Python and Kafka. Strong communication and leadership."


@pytest.mark.parametrize("resume, soft_points", [
    ("Python and Kafka. Clear communication and leadership.", 20.0),   # all hard skills, so all the soft points
    ("Python. Clear communication and leadership.", 10.0),             # half the hard skills, half the soft points
    ("Clear communication and leadership.", 0.0),                      # no hard skills, no soft points
])
def test_soft_skills_count_only_as_far_as_the_hard_skills_are_met(resume, soft_points):
    result = match_score(resume, SOFT_POSTING)

    assert result["match_out_of"] == {"hard": 80.0, "soft": 20.0}
    assert result["match_breakdown"]["soft"] == soft_points
    assert result["match_details"]["shown_before_scaling"]["soft"] == 1.0


def test_soft_skills_count_in_full_when_the_posting_names_no_hard_skills():
    result = match_score("Clear communication.", "Strong communication is needed.")

    assert result["match_breakdown"] == {"soft": 100.0}
