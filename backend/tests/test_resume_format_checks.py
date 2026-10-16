"""Whether an ATS can read the résumé: plain rules, shown beside the score."""
from backend.analyzer import resume_format_checks as format_checks

GOOD = """Priya Raman
priya@example.com | +1 415 555 0134 | Austin, TX
## Summary
Backend engineer.
## Experience
Senior Java Developer at Acme (Jan 2020 – Present — Austin, TX
- Built Spring Boot services on Kafka, cutting latency 30%
- Reviewed code for 6 engineers
## Skills
Java, Kafka
## Education
BS Computer Science — State University, 2016
""" + " word" * 260     # a normal one-to-two-page length


def _status(text: str) -> dict:
    return {check["id"]: check["status"] for check in format_checks.checks(text)["checks"]}


def test_a_well_formed_resume_passes_every_check():
    report = format_checks.checks(GOOD)
    assert report["passed"] == report["total"] == 9, report


def test_missing_contact_details_headings_and_dates_fail():
    bare = "Priya Raman\n## Work\nDeveloper at Acme\n- Did things\n"
    status = _status(bare)
    assert status["email"] == "fail" and status["phone"] == "fail"
    assert status["section-experience"] == status["section-education"] == status["section-skills"] == "fail"
    assert status["dates"] == "fail"


def test_a_date_range_is_not_a_phone_number():
    assert _status("Priya Raman\n2019 - 2022\n## Experience\n")["phone"] == "fail"


def test_bullets_without_numbers_and_icon_characters_are_flagged():
    text = GOOD.replace("cutting latency 30%", "cutting latency").replace("6 engineers", "six engineers") + ""
    status = _status(text)
    assert status["numbers"] == "fail" and status["icons"] == "fail"
