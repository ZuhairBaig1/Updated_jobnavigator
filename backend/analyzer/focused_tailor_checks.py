"""Whether the base résumé states what a sentence of the copy claims.

Used for every bullet, the summary and the skills of a focused or complete copy: a name or a number the résumé
does not state is unsupported, and an unsupported item in a list can be taken out of it.
"""
import re

from backend.analyzer import resume_schema as rs
from backend.analyzer.focused_tailor_basics import MOST_WORDS_IN_A_KEYWORD

# Only a list of at least this many items has an item that can be taken out.
FEWEST_ITEMS_IN_A_LIST = 3
# An item taken out of a list is one name plus at most this many words ("Kubernetes clusters").
MOST_WORDS_BESIDE_THE_NAME = 1

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#./-]*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# What separates the keywords of a requirement that names several: "Python, Pandas and FastAPI".
_KEYWORD_PARTS = re.compile(r",|;|\(|\)|\band\b|\bor\b")
# A skill's detail in brackets at its end: "AWS (EC2, S3)".
_DETAIL = re.compile(r"\s*\(([^()]*)\)\s*$")
# The letter-and-digit runs of a keyword, whatever joins them: "CI / CD" is "CI" and "CD".
_RUN = re.compile(r"[A-Za-z0-9+#.]+")


# ── What a sentence claims ──────────────────────────────────────────────────────

def unsupported_in(sentence: str, text: str, vocabulary: set, other_names: dict, prose: str | None = None) -> list:
    """The names a sentence states that `text` does not, and the numbers it states that `prose` does not, sorted.

    `prose` is the sentences the numbers may come from; it leaves out dates, whose digits would excuse any number.
    """
    names = _names_not_stated_in(sentence, text, vocabulary, other_names)
    numbers = _numbers_not_stated_in(sentence, text if prose is None else prose)
    return sorted(set(names + numbers))


def _names_not_stated_in(sentence: str, text: str, vocabulary: set, other_names: dict) -> list:
    # A phrase the text supports as a whole ("Test Driven Development" where it says "TDD") clears its words.
    cleared = set()
    for phrase in _capitalised_phrases_in(sentence):
        if _is_supported(phrase, text, other_names):
            cleared |= {rs._squash(word) for word in phrase.split()}
    not_stated = []
    for name in _names_in(sentence, vocabulary):
        if rs._squash(name) not in cleared and not _is_supported(name, text, other_names):
            not_stated.append(name)
    return not_stated


def _numbers_not_stated_in(sentence: str, source: str) -> list:
    stated = set(_NUMBER.findall(source))
    return [number for number in _NUMBER.findall(sentence or "") if number not in stated]


def _names_in(sentence: str, vocabulary: set) -> set:
    """The words of one sentence that read as the name of a tool or technique.

    That is an acronym, a name with a digit or inner capital (the patterns resume_schema already uses), a
    capitalised word that does not open the sentence, or a skill the résumé itself lists.
    """
    names = set()
    for pattern in rs._CLAIM_RES:
        names |= set(pattern.findall(sentence or ""))
    for word in _WORD.findall(sentence or "")[1:]:
        if word[0].isupper():
            names.add(word)
    for name in rs._named_tech(sentence, vocabulary):
        # A short lower-case word is prose, even when it happens to be the initials of a skill ("as" for "App Services").
        if not (name.islower() and name.isalpha() and len(name) <= 3):
            names.add(name)
    names = {name.strip(".,;:") for name in names}
    return {name for name in names if len(rs._squash(name)) >= 2}


def _capitalised_phrases_in(sentence: str) -> list:
    """Runs of two or more capitalised words after the sentence's first word: "Test Driven Development"."""
    phrases, run = [], []
    for word in _WORD.findall(sentence or "")[1:] + [""]:
        if word[:1].isupper():
            run.append(word.rstrip(".,"))
            continue
        if len(run) >= 2:
            phrases.append(" ".join(run))
        run = []
    return phrases


# ── Whether a text states a name ────────────────────────────────────────────────

def _is_supported(name: str, text: str, other_names: dict) -> bool:
    """Whether `text` states this name: as written, spelled out, with other word endings, by its initials,
    or under the résumé's own name for the same thing."""
    if _is_stated(name, text) or _is_spelled_out_in(name, text) or _is_stated_with_other_endings(name, text):
        return True
    if _initials_are_stated(name, text):
        return True
    return _is_stated_under_the_resumes_name(name, text, other_names)


def _is_stated(name: str, text: str) -> bool:
    """Whether `text` states this name. A short name must stand as a word of its own ("Go" is inside "MongoDB")."""
    key = rs._squash(name)
    if len(key) > 3:
        return key in rs._squash(text)
    return _stands_as_a_word(name, text)


def _stands_as_a_word(name: str, text: str) -> bool:
    """Whether `text` holds this name, in any capitals, with no letter or digit right before or after it."""
    pattern = r"(?<![A-Za-z0-9])" + re.escape(name) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text or "", re.IGNORECASE) is not None


def _is_spelled_out_in(acronym: str, text: str) -> bool:
    """Whether `text` writes this acronym out in words: "TDD" as "test-driven development"."""
    if not (2 <= len(acronym) <= 6 and acronym.isupper() and acronym.isalpha()):
        return False
    letters = acronym.lower()
    initials = "".join(word[0] for word in re.findall(r"[A-Za-z]+", (text or "").lower()))
    if letters not in initials:
        return False
    for start in range(len(initials) - len(letters) + 1):
        if initials.startswith(letters, start) and _starts_a_word_run(text, start, len(letters)):
            return True
    return False


def _starts_a_word_run(text: str, start: int, length: int) -> bool:
    """Whether the `length` words from word number `start` sit next to each other, joined only by spaces or hyphens."""
    spans = [match.span() for match in re.finditer(r"[A-Za-z]+", text or "")][start:start + length]
    for (_, end_of_word), (start_of_next, _) in zip(spans, spans[1:]):
        if not set(text[end_of_word:start_of_next]) <= {" ", "-"}:
            return False
    return True


def _is_stated_with_other_endings(phrase: str, text: str) -> bool:
    """Whether `text` states a phrase of several words allowing each word another ending: "REST APIs" where it says "RESTful APIs"."""
    words = _WORD.findall(phrase or "")
    if len(words) < 2:
        return False
    stems = [_word_stem(word) + r"\w*" for word in words]
    return re.search(r"(?<![A-Za-z0-9])" + r"[\s/\-]+".join(stems), text or "", re.IGNORECASE) is not None


def _word_stem(word: str) -> str:
    """The word without a plural s (a word over three letters), escaped for a pattern."""
    return re.escape(word.rstrip("s") if len(word) > 3 else word)


def _initials_are_stated(name: str, text: str) -> bool:
    """Whether `text` states a name of several words by its initials: "Azure Data Factory" as "ADF"."""
    words = _WORD.findall(name)
    if len(words) < 2:
        return False
    initials = "".join(word[0] for word in words)
    return _stands_as_a_word(initials, text)


def _is_stated_under_the_resumes_name(name: str, text: str, other_names: dict) -> bool:
    """Whether `name` is the posting's name for something `text` states under the résumé's own name."""
    for posting_name, resume_name in other_names.items():
        if _is_stated(name, posting_name) and _is_stated(resume_name, text):
            return True
    return False


def _adds_words_to(longer: str, shorter: str) -> bool:
    """Whether `longer` is `shorter` with words added: "GitHub Actions" to "GitHub". That names a narrower thing."""
    long_words = [rs._squash(word) for word in _WORD.findall(longer or "")]
    short_words = [rs._squash(word) for word in _WORD.findall(shorter or "")]
    if not short_words or len(long_words) <= len(short_words):
        return False
    width = len(short_words)
    for start in range(len(long_words) - width + 1):
        if long_words[start:start + width] == short_words:
            return True
    return False


# ── Whether a text names a keyword, however it is spelled ───────────────────────

def _names_the_requirement(text: str, requirement: str) -> bool:
    """Whether a text names a requirement, or one of the keywords a longer requirement is made of, however it is spelled.

    "React.js and modern frontend development practices" is named by a bullet that says "React.js".
    """
    for part in [requirement] + _KEYWORD_PARTS.split(requirement or ""):
        part = part.strip(" .:")
        if not part or len(part.split()) > MOST_WORDS_IN_A_KEYWORD or len(rs._squash(part)) < 2:
            continue
        if _however_it_is_spelled(part).search(text or ""):
            return True
    return False


def _however_it_is_spelled(keyword: str):
    """A pattern for the keyword whatever its capitals, and whether its parts are joined by a space, a hyphen or a slash."""
    runs = _RUN.findall(keyword)
    return re.compile(r"(?<![A-Za-z0-9])" + r"[\s/\-]*".join(re.escape(run) for run in runs) + r"(?![A-Za-z0-9+#])",
                      re.IGNORECASE)


# ── Taking an unsupported item out of a list ────────────────────────────────────

def _with_the_unsupported_items_taken_out(sentence: str, unsupported: list, allowed_text: str, vocabulary: set,
                                          other_names: dict, prose: str) -> str:
    """The sentence without the list items naming what it should not, if what is left states nothing unsupported; else ""."""
    shorter = _without_the_list_items_naming(sentence, unsupported)
    if shorter and not unsupported_in(shorter, allowed_text, vocabulary, other_names, prose=prose):
        return shorter
    return ""


def _without_the_list_items_naming(sentence: str, names: list) -> str:
    """A sentence that lists things, without the items that name any of `names`; "" when it cannot be done cleanly.

    "Skilled in Java, NoSQL, and Kafka." without NoSQL is "Skilled in Java, and Kafka."
    """
    items = re.split(r",\s*", sentence.rstrip())
    if len(items) < FEWEST_ITEMS_IN_A_LIST:
        return ""
    goes = [any(_is_stated(name, item) for name in names) for item in items]
    kept = [item for item, item_goes in zip(items, goes) if not item_goes]
    # Nothing to take out, too little left, or the opening of the sentence itself is the trouble.
    if len(kept) < 2 or len(kept) == len(items) or goes[0]:
        return ""
    # Only a short item of a list may go. A longer one is a clause carrying the rest of the sentence
    # ("and microservices, supporting patient workflows"), and taking it out leaves nonsense.
    taken_out = [re.sub(r"^(and|or)\s+", "", item.rstrip(".")) for item, item_goes in zip(items, goes) if item_goes]
    if not all(_is_little_more_than(item, names) for item in taken_out):
        return ""
    # An item that closes a list in the middle of a sentence ("Java, Spring Boot, and X, supporting ...") takes
    # the list's "and" with it, and what is left does not read; the sentence is not cut then.
    last_position = len(items) - 1
    for position, (item, item_goes) in enumerate(zip(items, goes)):
        if item_goes and position != last_position and re.match(r"(and|or)\s", item):
            return ""
    # When the last item went, the new last item takes the "and".
    if goes[-1]:
        last = kept[-1].rstrip(".")
        kept[-1] = (last if last.startswith("and ") else f"and {last}") + "."
    return ", ".join(kept)


def _is_little_more_than(item: str, names: list) -> bool:
    """Whether a list item is just one of the names, give or take a word ("Kubernetes clusters"), and not a clause."""
    for name in names:
        if _is_stated(name, item) and len(item.split()) <= len(name.split()) + MOST_WORDS_BESIDE_THE_NAME:
            return True
    return False
