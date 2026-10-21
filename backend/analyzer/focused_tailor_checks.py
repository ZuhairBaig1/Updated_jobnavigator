"""Whether the base résumé states what a sentence of the copy claims.

Used for every bullet, the summary and the skills of a focused or complete copy: a name or a number the résumé
does not state is unsupported, and an unsupported item in a list can be taken out of it.

A name is a tool, a technique or a skill: one the skills dictionary knows (not a personal trait), an acronym, a name
with a digit or an inner capital, or a skill the résumé lists. An ordinary word the posting happens to capitalise
("Architecture", "Engines", "IDs") claims nothing. A name is stated when the text writes it, spells it out, writes it
with other word endings ("containerized" for Containerization) or in other words of its own ("microservices" and
"architecture" for Microservices Architecture), names the same tool another way (React for React.js, EMR for AWS
EMR), names a member of its family (MongoDB for NoSQL), or states it with only a word of use after it (CI/CD for "CI/CD
tools") — but never by a narrower thing ("GitHub Actions" is not GitHub, "Spring Security" is not Spring, "Scrum" is
not Agile), and a pair ("Linux/Unix") needs both its halves.

Measured on 16 copies, the same replies through the old checks and these, each removal judged by Claude Sonnet 5.5:
the old checks made 21 truth removals, only 8 of them real, and dropped 29 skills, 25 of them stated; these made 8
truth removals, all real, and dropped 5 skills, 2 of them real and 3 borderline (2026-10-21).
"""
import re

from backend.analyzer import keyword_match
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
# Endings a word takes for the same thing, longest first: "containerized" and "containerization" are both "container".
_ENDINGS = ("ization", "isation", "ations", "ation", "izing", "ising", "ized", "ised", "ings", "ing", "ed", "es", "s")
# A shorter stem matches too many words to be the same thing.
SHORTEST_STEM = 5
# Words that join the parts of a name and claim nothing of their own: "Containerization-based".
_JOINING_WORDS = {"and", "as", "based", "for", "in", "of", "on", "the", "to", "with"}
# Words that may follow a tool's name without making a narrower product of it: "Microservices Architecture", "ETL
# pipelines". Any other capitalised word after a tool names a product of its own ("Spring Security", "Kafka Streams").
_GENERIC_WORDS = {"application", "applications", "architecture", "architectures", "cloud", "containerization",
                  "data", "deployment", "deployments", "design", "development", "driven", "environment", "environments",
                  "framework", "frameworks", "integration", "integrations", "pipeline", "pipelines", "platform",
                  "platforms", "practices", "processes", "scripting", "service", "services", "solutions", "system",
                  "systems", "tools", "workflows"}
# Generic words that say how a named tool is used and never finish a product's name: "CI/CD tools", "Python
# Scripting". "Cloud", "data", "pipelines" and "integration" do ("Spring Cloud", "Spring Data", "Azure Pipelines").
_WORDS_OF_USE = {"architecture", "architectures", "development", "environment", "environments", "practices",
                 "processes", "scripting", "services", "solutions", "system", "systems", "tools", "workflows"}
# A family the résumé names by naming one of its members (the skills dictionary's standard names): "NoSQL databases"
# where it lists MongoDB. The words of _FAMILY_FILLERS are left out of a family's name.
_RELATIONAL = {"PostgreSQL", "MySQL", "Oracle", "SQL Server", "mariadb", "sqlite", "amazon-aurora"}
_FAMILIES = {"nosql": {"MongoDB", "DynamoDB", "Cassandra", "Redis", "Elasticsearch", "google-cloud-firestore", "hbase",
                       "couchbase", "neo4j", "Azure Cosmos DB"},
             "relational": _RELATIONAL, "rdbms": _RELATIONAL}
_FAMILY_FILLERS = {"database", "databases", "db", "dbs", "management", "system", "systems"}


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
    """The words of one sentence that read as the name of a tool, technique or skill.

    That is an acronym, a name with a digit or inner capital (the patterns resume_schema already uses), a tool or
    technique the skills dictionary knows (a personal trait is not checked here, and a technique only when written
    as a name, since "monitoring" in prose is an activity), or a skill the résumé itself lists.
    """
    names = set()
    for pattern in rs._CLAIM_RES:
        names |= set(pattern.findall(sentence or ""))
    for mention in keyword_match.mentions_in(sentence or ""):
        written = mention["as_written"]
        if mention["type"] == keyword_match.HARD_SKILL or (mention["type"] == keyword_match.OTHER_KEYWORD
                                                            and written[:1].isupper()):
            names.add(written)
    for name in rs._named_tech(sentence, vocabulary):
        # A short lower-case word is prose, even when it happens to be the initials of a skill ("as" for "App Services").
        if not (name.islower() and name.isalpha() and len(name) <= 3):
            names.add(name)
    names = {name.strip(".,;:") for name in names}
    return {name for name in names if len(rs._squash(name)) >= 2}


def _capitalised_phrases_in(sentence: str) -> list:
    """Runs of two or more capitalised words after the sentence's first word: "Test Driven Development".

    A comma or other punctuation ends a run, so each item of a list is a phrase of its own.
    """
    phrases = []
    for number, piece in enumerate(re.split(r"[,;:()]", sentence or "")):
        run = []
        for word in _WORD.findall(piece)[1 if number == 0 else 0:] + [""]:
            if word[:1].isupper():
                run.append(word.rstrip(".,"))
                continue
            if len(run) >= 2:
                phrases.append(" ".join(run))
            run = []
    return phrases


# ── Whether a text states a name ────────────────────────────────────────────────

def _is_supported(name: str, text: str, other_names: dict) -> bool:
    """Whether `text` states this name: as written, spelled out, with other word endings, by its initials, in other
    words of its own, as the same tool named another way, by a member of its family, as a stated name with a generic
    word after it, or under the résumé's own name for the same thing. A pair ("Linux/Unix") needs both its halves."""
    if _is_stated(name, text):
        return True
    halves = _halves_of_a_pair(name)
    if halves:
        return all(_is_supported(half, text, other_names) for half in halves)
    if _is_spelled_out_in(name, text) or _is_stated_with_other_endings(name, text):
        return True
    if _initials_are_stated(name, text) or _is_stated_with_another_ending(name, text):
        return True
    if _each_of_its_words_is_stated(name, text) or _is_the_same_tool_named_another_way(name, text):
        return True
    if _a_member_of_its_family_is_named(name, text) or _is_a_stated_name_with_a_generic_word(name, text, other_names):
        return True
    return _is_stated_under_the_resumes_name(name, text, other_names)


def _halves_of_a_pair(name: str) -> list:
    """The two or more names a pair joins with slashes: "Linux/Unix" is Linux and Unix. [] for anything else."""
    if "/" not in name or " " in name.strip():
        return []
    halves = [half for half in name.strip().split("/")]
    return halves if all(len(half) >= 2 for half in halves) else []


def _stem(word: str) -> str:
    """The word without the ending that makes another word of the same thing: "containerized" is "container"."""
    lower = word.lower()
    for ending in _ENDINGS:
        if lower.endswith(ending) and len(lower) - len(ending) >= SHORTEST_STEM:
            return lower[:-len(ending)]
    return lower


def _is_stated_with_another_ending(word: str, text: str) -> bool:
    """Whether `text` states one word with another ending: "Containerization" where it says "containerized"."""
    if not word.isalpha() or len(_stem(word)) < SHORTEST_STEM:
        return False
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(_stem(word)), text or "", re.IGNORECASE) is not None


def _each_of_its_words_is_stated(name: str, text: str) -> bool:
    """Whether `text` states every word of a name of several, in any order and with any ending: "Microservices
    Architecture" where it says "microservices" and "architecture", "Docker-containerized" where it says Docker and
    containerized. Never for a name the skills dictionary knows as one thing ("GitHub Actions" is not GitHub and
    actions), nor for a tool followed by a product's own name ("Spring Security" is not Spring and security)."""
    parts = [part for part in re.split(r"[\s/\-]+", name.strip()) if part]
    words = [part for part in parts if len(part) >= 3 and part.lower() not in _JOINING_WORDS]
    if len(parts) < 2 or not words or _the_dictionary_knows(name) or _names_a_product(parts):
        return False
    return all(_is_stated(word, text) or _is_stated_with_another_ending(word, text) for word in words)


def _the_dictionary_knows(name: str) -> bool:
    return bool(keyword_match._lookup(name.split()) or keyword_match._lookup(re.split(r"[\s\-]+", name)))


def _names_a_product(parts: list) -> bool:
    """Whether the words name a product of a tool: a tool's name followed by a capitalised word that is not generic."""
    for position, part in enumerate(parts[:-1]):
        entry = keyword_match._lookup([part])
        if entry and keyword_match.KEYWORD_TYPES[entry[2]] == keyword_match.HARD_SKILL:
            following = parts[position + 1]
            if following[:1].isupper() and following.lower() not in _GENERIC_WORDS:
                return True
    return False


def _is_the_same_tool_named_another_way(name: str, text: str) -> bool:
    """Whether `text` names the same tool another way, as the skills dictionary knows it: "React" for React.js, "EMR"
    for AWS EMR.

    Only when both names start with the same word, or the name holds the text's own words for it: the dictionary also
    files a narrower thing under a broader name ("Scrum" under Agile), and that is not the same thing.
    """
    entry = keyword_match._lookup(name.split())
    if not entry:
        return False
    standard = entry[0]
    for mention in keyword_match.mentions_in(text or ""):
        if mention["standard"] == standard and (_first_word(name) == _first_word(standard)
                                                or _stands_as_a_word(mention["as_written"], name)):
            return True
    return False


def _first_word(name: str) -> str:
    return re.split(r"[\s.\-]+", name.lower())[0]


def _a_member_of_its_family_is_named(name: str, text: str) -> bool:
    """Whether the name is a family (NoSQL, relational databases) and `text` names one of its members."""
    words = [word for word in re.split(r"[\s\-]+", name.lower().strip()) if word not in _FAMILY_FILLERS]
    members = _FAMILIES.get(" ".join(words))
    return bool(members) and any(mention["standard"] in members for mention in keyword_match.mentions_in(text or ""))


def _is_a_stated_name_with_a_generic_word(name: str, text: str, other_names: dict) -> bool:
    """Whether the name is a stated name, or a stated word, with generic words after it, which claim nothing more:
    "CI/CD tools" where `text` says CI/CD, "Python Scripting" where it says Python, "distributed systems" where it says
    "distributed services"."""
    words = name.split()
    while words and words[-1].lower() in _GENERIC_WORDS:
        words.pop()
    added = [word.lower() for word in name.split()[len(words):]]
    if not words or not added:
        return False
    rest = " ".join(words)
    if keyword_match._lookup(words) or any(pattern.fullmatch(rest) for pattern in rs._CLAIM_RES):
        # a tool's name: only words of use may follow it, and it must be stated in any of the ways above
        return set(added) <= _WORDS_OF_USE and _is_supported(rest, text, other_names)
    # an ordinary word: the text must write it before a generic word of its own
    generic = "|".join(sorted(_GENERIC_WORDS))
    return re.search(r"(?<![A-Za-z0-9])" + re.escape(rest) + r"\s+(?:" + generic + r")\b", text or "", re.IGNORECASE) is not None


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
