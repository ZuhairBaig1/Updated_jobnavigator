"""The names, numbers and small résumé helpers that every part of the focused and complete tailoring uses."""

# How much the posting wants a thing: it requires it, would like it, or lists it as part of the work.
MUST_HAVE, NICE_TO_HAVE, DUTY = "must", "nice", "duty"

# How many bullets a focused copy gives each role, most recent first: five, then four, then three for every earlier one.
BULLETS_OF_THE_LATEST_ROLES = (5, 4)
BULLETS_OF_AN_EARLIER_ROLE = 3

# A keyword is a name of a few words; a longer requirement is a sentence, not one keyword.
MOST_WORDS_IN_A_KEYWORD = 4


def _roles(base: dict) -> list:
    return [role for role in (base.get("experience") or []) if isinstance(role, dict)]


def _projects(base: dict) -> list:
    return [project for project in (base.get("projects") or []) if isinstance(project, dict)]


def _bullets_of(entry: dict) -> list:
    """A role's or project's bullets, the ones that are text."""
    return [bullet for bullet in entry.get("bullets") or [] if isinstance(bullet, str)]
