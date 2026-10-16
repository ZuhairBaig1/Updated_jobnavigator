"""Whether an ATS can read the résumé: plain rules, no model. Shown beside the ATS score, never inside it.

These describe the résumé rather than its fit to a posting, so a well-made résumé can still be wrong for the
job (ATS scoring rules 10.6, A10.16). Every check says what it found; there is no hidden weighting (A6.8).
"""
import re

from backend.analyzer import ats_text

PASS, WARN, FAIL = "pass", "warn", "fail"
STANDARD_HEADINGS = ("experience", "education", "skills")
FEWEST_WORDS, MOST_WORDS = 250, 1100            # about one to two pages
SHARE_OF_BULLETS_WITH_A_NUMBER = 0.5            # "quantify at least half the bullets"
A_NUMBER = re.compile(r"\d|%|\$|£|€")
# Icon fonts draw from the Private Use Area; a parser turns those characters into question marks.
AN_ICON_CHARACTER = re.compile("[-]")


def checks(resume_text: str) -> dict:
    """{"passed", "total", "checks": [{"id", "label", "status", "detail"}]}."""
    found = [
        _email(resume_text),
        _phone(resume_text),
        *[_heading(resume_text, name) for name in STANDARD_HEADINGS],
        _dates_on_each_role(resume_text),
        _length(resume_text),
        _bullets_with_a_number(resume_text),
        _icon_characters(resume_text),
    ]
    return {"passed": sum(check["status"] == PASS for check in found), "total": len(found), "checks": found}


def _check(check_id: str, label: str, status: str, detail: str) -> dict:
    return {"id": check_id, "label": label, "status": status, "detail": detail}


def _top_lines(resume_text: str) -> str:
    """The lines before the first section: where the name and contact details are printed."""
    top = []
    for line in (resume_text or "").splitlines():
        if line.startswith("## "):
            break
        top.append(line)
    return "\n".join(top)


def _email(resume_text: str) -> dict:
    found = ats_text.AN_EMAIL.search(resume_text or "")
    return _check("email", "Email address", PASS if found else FAIL,
                  "Found in the body text." if found else "No email address in the text an ATS reads.")


def _phone(resume_text: str) -> dict:
    numbers = [match.group(0) for match in ats_text.A_PHONE_NUMBER.finditer(_top_lines(resume_text))]
    found = any(not ats_text.looks_like_a_range(number) for number in numbers)
    return _check("phone", "Phone number", PASS if found else FAIL,
                  "Found." if found else "No phone number near the top of the résumé.")


def _heading(resume_text: str, name: str) -> dict:
    found = any(heading == name for heading, _lines in ats_text.sections(resume_text))
    return _check(f"section-{name}", f"\"{name.capitalize()}\" section heading", PASS if found else FAIL,
                  "Found under a heading parsers recognise." if found
                  else f"No \"{name.capitalize()}\" heading; a parser may not find this part of the résumé.")


def _dates_on_each_role(resume_text: str) -> dict:
    roles = ats_text.roles_in(resume_text)
    if not roles:
        return _check("dates", "Dates on each role", FAIL, "No roles found under Experience.")
    dated = sum(bool(role["months"]) for role in roles)
    status = PASS if dated == len(roles) else WARN if dated else FAIL
    return _check("dates", "Dates on each role", status, f"{dated} of {len(roles)} roles state their dates.")


def _length(resume_text: str) -> dict:
    words = len(re.findall(r"\w+", resume_text or ""))
    if words < FEWEST_WORDS:
        return _check("length", "Length", WARN, f"{words} words — short for most roles.")
    if words > MOST_WORDS:
        return _check("length", "Length", WARN, f"{words} words — longer than two pages.")
    return _check("length", "Length", PASS, f"{words} words — a normal one-to-two-page length.")


def _bullets_with_a_number(resume_text: str) -> dict:
    bullets = [line.strip().lstrip("-•* ") for heading, lines in ats_text.sections(resume_text)
               if heading in ("experience", "projects") for line in lines if line.strip().startswith("-")]
    if not bullets:
        return _check("numbers", "Bullets with a number", FAIL, "No achievement bullets under Experience or Projects.")
    with_a_number = sum(bool(A_NUMBER.search(bullet)) for bullet in bullets)
    share = with_a_number / len(bullets)
    status = PASS if share >= SHARE_OF_BULLETS_WITH_A_NUMBER else WARN if with_a_number else FAIL
    return _check("numbers", "Bullets with a number", status,
                  f"{with_a_number} of {len(bullets)} bullets carry a number, percentage or amount.")


def _icon_characters(resume_text: str) -> dict:
    found = AN_ICON_CHARACTER.findall(resume_text or "")
    return _check("icons", "No icon-font characters", FAIL if found else PASS,
                  f"{len(found)} icon characters a parser would turn into question marks." if found else "None found.")
