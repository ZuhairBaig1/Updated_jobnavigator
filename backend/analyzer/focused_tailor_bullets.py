"""Each role's bullets in a focused or complete copy: checked against the résumé, then chosen, swapped or completed.

1. Every rewritten bullet is checked against its own role (step 2 of build_focused_resume).
2. A focused copy keeps a set number of bullets a role. The model writes exactly that many and chooses them itself;
   the choosing here only acts when a role ends up with too few (the role's own bullets fill in) or too many.
3. A complete copy keeps every other bullet of the role below the rewritten ones.
4. A focused copy then swaps in a résumé bullet for each must-have the résumé's bullets name and no bullet of the copy
   does (step 4). Nothing else is swapped: the model's choice stands.
"""
import difflib
import logging
import re

from backend.analyzer import keyword_match
from backend.analyzer import resume_schema as rs
from backend.analyzer.focused_tailor_basics import BULLETS_OF_AN_EARLIER_ROLE, BULLETS_OF_THE_LATEST_ROLES, MUST_HAVE
from backend.analyzer.focused_tailor_checks import (_WORD, _names_the_requirement, _with_the_unsupported_items_taken_out,
                                                    _word_stem, unsupported_in)

logger = logging.getLogger("jobnavigator.focused_tailor")

# Opening verbs that claim more than doing the work: who conceived it, or who was in charge of it.
VERBS_THAT_CLAIM_MORE = ("designed", "architected", "led", "owned", "spearheaded", "headed", "managed", "directed",
                         "drove", "pioneered", "conceived", "oversaw")
# How alike a rewrite and a résumé bullet must be (0 to 1) for that bullet to stand in as its source.
SIMILARITY_OF_THE_CLOSEST_SOURCE = 0.45

# Points for a bullet that shows a requirement, when choosing between bullets.
WORTH_OF_SHOWING = {"must": 3, "nice": 2, "duty": 1}
# Extra points for a bullet that states a measured result.
WORTH_OF_A_MEASURED_RESULT = 2
# A measured result: a percentage (40%), an amount of money ($2M) or a multiple (3x).
_A_MEASURED_RESULT = re.compile(r"\d+(?:\.\d+)?\s*%|[$€£]\s?\d|\b\d+(?:\.\d+)?x\b")

# A role's first bullets are the ones read, and the reply put its strongest there, so a swap never takes their place.
BULLETS_NEVER_SWAPPED_OUT = 2

# A complete copy does not print a résumé bullet again when a rewrite already holds this share of its words.
SHARE_OF_WORDS_THAT_MAKES_A_REPEAT = 0.8


# ── 1. Checking each rewritten bullet ───────────────────────────────────────────

def focused_bullets(reply_bullets: list, base_bullets: list, allowed_text: str, vocabulary: set, other_names: dict) -> list:
    """A role's or project's rewritten bullets, each kept, trimmed, or put back to the résumé's own sentence.

    Each is {"text", "from": the numbers of its source bullets, "as_written": whether it went back to the résumé's sentence}.
    """
    kept = []
    for bullet in reply_bullets or []:
        if not isinstance(bullet, dict):
            continue
        checked = _checked_bullet(bullet, base_bullets, allowed_text, vocabulary, other_names)
        if checked and checked["text"] not in [kept_bullet["text"] for kept_bullet in kept]:
            kept.append(checked)
    return kept


def _checked_bullet(bullet: dict, base_bullets: list, allowed_text: str, vocabulary: set, other_names: dict) -> dict | None:
    """One rewritten bullet as its role supports it; None when it has no text or no source."""
    text = rs._text(bullet.get("text"))
    sources = _numbers_of_its_sources(bullet, base_bullets)
    if text and not sources:
        sources = _closest_source(text, base_bullets)
    if not text or not sources:
        return None

    prose = " ".join(base_bullets)
    unsupported = _what_it_states_beyond_its_role(text, sources, base_bullets, allowed_text, vocabulary, other_names, prose)
    as_written = False
    if unsupported:
        # One unsupported item in a list costs that item, not the whole rewrite and the keywords it carries.
        shorter = _with_the_unsupported_items_taken_out(text, unsupported, allowed_text, vocabulary, other_names, prose)
        if shorter:
            logger.warning("Focused tailor: a bullet listed %s, which its role does not — taken out of the list",
                           ", ".join(unsupported[:6]))
            text = shorter
        else:
            logger.warning("Focused tailor: a bullet stated %s, which its role does not — keeping the résumé's own sentence",
                           ", ".join(unsupported[:6]))
            text, sources, as_written = base_bullets[sources[0] - 1], sources[:1], True

    source_sentences = [base_bullets[number - 1] for number in sources]
    text = with_the_verb_its_sources_use(text, source_sentences)
    return {"text": text, "from": sources, "as_written": as_written}


def _numbers_of_its_sources(bullet: dict, base_bullets: list) -> list:
    """The numbers of the base bullets a rewrite says it was written from, in order; ones the role lacks are dropped."""
    numbers = bullet.get("from_bullets") if isinstance(bullet.get("from_bullets"), list) else []
    return [number for number in numbers if isinstance(number, int) and 1 <= number <= len(base_bullets)]


def _closest_source(text: str, base_bullets: list) -> list:
    """When the reply named no source: [the number of the role's bullet most like the rewrite], or [] when none is alike."""
    closest = difflib.get_close_matches(text, base_bullets, 1, SIMILARITY_OF_THE_CLOSEST_SOURCE)
    return [base_bullets.index(closest[0]) + 1] if closest else []


def _what_it_states_beyond_its_role(text: str, sources: list, base_bullets: list, allowed_text: str, vocabulary: set,
                                    other_names: dict, prose: str) -> list:
    """What a bullet names that its role does not state, except a phrase whose every word its own sources state."""
    unsupported = unsupported_in(text, allowed_text, vocabulary, other_names, prose=prose)
    its_sources = " ".join(base_bullets[number - 1] for number in sources)
    return [name for name in unsupported if not _every_word_is_stated_in(name, its_sources)]


def with_the_verb_its_sources_use(text: str, sources: list) -> str:
    """The bullet, opening as its first source does when it opened by claiming more than any source says.

    "Designed cloud-native applications" written from "Built cloud-native applications" goes back to "Built".
    A bullet whose sources do use the verb, or one that opens another way, is left alone.
    """
    words = text.split(maxsplit=1)
    source_words = (sources[0] if sources else "").split(maxsplit=1)
    if len(words) < 2 or not source_words:
        return text
    opening = words[0].strip(",").lower()
    said_by_a_source = any(re.search(r"\b" + re.escape(opening[:-2]), source, re.IGNORECASE) for source in sources)
    if opening not in VERBS_THAT_CLAIM_MORE or said_by_a_source:
        return text
    return f"{source_words[0].strip(',')} {words[1]}"


def _every_word_is_stated_in(phrase: str, sources: str) -> bool:
    """Whether a bullet's own sources state every word of a phrase, each allowed another ending.

    "REST APIs" is nothing new in a bullet written from "Built RESTful services and public APIs". Only the bullet's
    own sources count, not the whole role, so two words from unrelated bullets cannot add up to a product name.
    """
    words = _WORD.findall(phrase or "")
    if len(words) < 2:
        return False
    for word in words:
        if not re.search(r"(?<![A-Za-z0-9])" + _word_stem(word) + r"\w*", sources or "", re.IGNORECASE):
            return False
    return True


# ── 2. A focused copy: choosing a role's bullets ────────────────────────────────

def bullets_a_role_gets(role_number: int) -> int:
    """How many bullets a focused copy gives a role; role 1 is the most recent."""
    if role_number <= len(BULLETS_OF_THE_LATEST_ROLES):
        return BULLETS_OF_THE_LATEST_ROLES[role_number - 1]
    return BULLETS_OF_AN_EARLIER_ROLE


def the_most_important(bullets: list, base_bullets: list, how_many: int, role_number: int, requirements: list) -> list:
    """Exactly `how_many` bullets for a role (fewer only when the résumé has fewer), in the order they stood.

    The model chooses: when it wrote too many, its first ones stay, since it puts the ones that matter most first.
    When it wrote too few, they all stay and the role's unused bullets fill the gap, picked one at a time, each time
    the one worth the most for what no picked bullet shows yet, as the résumé wrote them.
    """
    if len(bullets) >= how_many:
        return list(bullets[:how_many])
    candidates = list(bullets) + _unused_bullets_of_the_role(bullets, base_bullets)
    picked = list(range(len(bullets)))
    already_shown = set()
    for position in picked:
        already_shown |= _shown_by(candidates[position], role_number, requirements)

    while len(picked) < min(how_many, len(candidates)):
        best = _worth_adding_most(candidates, picked, already_shown, role_number, requirements)
        picked.append(best)
        already_shown |= _shown_by(candidates[best], role_number, requirements)
    return [candidates[position] for position in sorted(picked)]


def _unused_bullets_of_the_role(bullets: list, base_bullets: list) -> list:
    """The role's own bullets that no rewrite was written from, as the résumé wrote them."""
    used = {number for bullet in bullets for number in bullet["from"]}
    texts = [bullet["text"] for bullet in bullets]
    unused = []
    for number, text in enumerate(base_bullets, start=1):
        if number not in used and text not in texts:
            unused.append({"text": text, "from": [number], "as_written": True})
    return unused


def _worth_adding_most(candidates: list, picked: list, already_shown: set, role_number: int, requirements: list) -> int:
    """The position of the unpicked candidate that adds the most points; on a tie, the earliest."""
    best, best_points = None, None
    for position, candidate in enumerate(candidates):
        if position in picked:
            continue
        points = _points_for_adding(candidate, already_shown, role_number, requirements)
        if best is None or points > best_points:
            best, best_points = position, points
    return best


def _points_for_adding(bullet: dict, already_shown: set, role_number: int, requirements: list) -> int:
    """Points for each requirement the bullet shows that no picked bullet shows yet, plus its measured result's."""
    newly_shown = _shown_by(bullet, role_number, requirements) - already_shown
    measured_result = WORTH_OF_A_MEASURED_RESULT if _A_MEASURED_RESULT.search(bullet["text"]) else 0
    return _points_for(newly_shown, requirements) + measured_result


def _shown_by(bullet: dict, role_number: int, requirements: list) -> set:
    """The positions of the requirements a bullet shows: the ones it names, and the ones the reply matched to its sources."""
    shown = set()
    for position, requirement in enumerate(requirements):
        if (_names_the_requirement(bullet["text"], requirement["requirement"])
                or _matched_to_one_of_its_sources(bullet, role_number, requirement)):
            shown.add(position)
    return shown


def _matched_to_one_of_its_sources(bullet: dict, role_number: int, requirement: dict) -> bool:
    """Whether the reply pointed this requirement at one of the bullets this bullet was written from."""
    for pointed_role, pointed_bullet in requirement["shown_by"]:
        if pointed_role == role_number and pointed_bullet in bullet["from"]:
            return True
    return False


def _points_for(requirement_positions, requirements: list) -> int:
    """The points WORTH_OF_SHOWING gives for showing these requirements."""
    return sum(WORTH_OF_SHOWING.get(requirements[position]["importance"], 0) for position in requirement_positions)


# ── 3. A complete copy: keeping every other bullet ──────────────────────────────

def _with_all_the_rest(bullets: list, order_asked_for, base_bullets: list) -> list:
    """The rewritten bullets, then every bullet of the role none of them was written from, as the résumé wrote it.

    Those follow in the order the reply asked for; any it forgot come last, in the résumé's own order. Nothing is lost.
    A bullet a rewrite already says nearly all of is not printed again: the reply sometimes names the wrong source.
    """
    covered = {number for bullet in bullets for number in bullet["from"]}
    rewritten_texts = [bullet["text"] for bullet in bullets]
    asked_for = [number for number in order_asked_for or []
                 if isinstance(number, int) and 1 <= number <= len(base_bullets)]
    every_number = list(range(1, len(base_bullets) + 1))
    all_bullets = list(bullets)
    for number in asked_for + every_number:
        if number in covered:
            continue
        covered.add(number)
        text = base_bullets[number - 1]
        if text in [bullet["text"] for bullet in all_bullets]:
            continue
        if any(_says_nearly_all_of(rewrite, text) for rewrite in rewritten_texts):
            continue
        all_bullets.append({"text": text, "from": [number], "as_written": False})
    return all_bullets


def _says_nearly_all_of(rewrite: str, original: str) -> bool:
    """Whether a rewritten bullet already holds nearly every content word of an original one."""
    words_of_the_original = _content_words(original)
    if not words_of_the_original:
        return False
    share_held = len(words_of_the_original & _content_words(rewrite)) / len(words_of_the_original)
    return share_held >= SHARE_OF_WORDS_THAT_MAKES_A_REPEAT


def _content_words(text: str) -> set:
    """The words of four characters or more, lower-cased and without a plural s."""
    return {word.lower().rstrip("s") for word in re.findall(r"[A-Za-z][A-Za-z0-9+#.]{3,}", text or "")}


# ── 4. A focused copy: the must-haves the model left out ────────────────────────

def swapped_in_for_missing_musts(bullets_of_each_role: list, roles: list, requirements: list) -> int:
    """For each must-have no chosen bullet names, swaps in a résumé bullet that names it, as written; returns how many.

    The safety net under the model's own choice: it acts only on a must-have that names a tool or technology the skills
    dictionary knows ("SQL", "Kubernetes"), and that the résumé's bullets name word for word. A degree's field, a duty
    or a trait is not one: "Engineering" once swapped in a bullet only for saying "engineering efficiency".
    The most recent role with such a bullet takes it, in the slot whose loss costs the copy the fewest points, never in
    place of a role's first bullets. A screener counts a core skill only when a role shows it, so this is not limited to
    one swap a role. Changes bullets_of_each_role in place.
    """
    swapped = 0
    for requirement in requirements:
        keyword = requirement["requirement"]
        if (requirement["importance"] != MUST_HAVE or not _names_a_tool(keyword)
                or _named_anywhere(bullets_of_each_role, keyword)):
            continue
        for role_number, role in enumerate(roles, start=1):
            chosen = bullets_of_each_role[role_number - 1]
            naming_bullet = _unused_bullet_naming(keyword, role, chosen)
            if naming_bullet is None or len(chosen) <= BULLETS_NEVER_SWAPPED_OUT:
                continue
            slot = _cheapest_slot_to_give_up(bullets_of_each_role, role_number, naming_bullet, requirements)
            if slot is None:
                continue   # every slot would leave another must-have unnamed
            chosen[slot] = naming_bullet
            swapped += 1
            break
    return swapped


def _names_a_tool(keyword: str) -> bool:
    """Whether a requirement's name is a tool or technology the skills dictionary knows (a hard skill)."""
    entry = keyword_match._lookup(keyword.split())
    return bool(entry) and keyword_match.KEYWORD_TYPES[entry[2]] == keyword_match.HARD_SKILL


def _unused_bullet_naming(keyword: str, role: dict, chosen: list) -> dict | None:
    """The role's first bullet that names the keyword and that no chosen bullet was written from, as the résumé wrote it."""
    used = {number for bullet in chosen for number in bullet["from"]}
    for number, text in enumerate(role.get("bullets") or [], start=1):
        if isinstance(text, str) and _names_the_requirement(text, keyword) and number not in used:
            return {"text": text, "from": [number], "as_written": True}
    return None


def _cheapest_slot_to_give_up(bullets_of_each_role: list, role_number: int, new_bullet: dict, requirements: list) -> int | None:
    """The slot of the role (never one of its first bullets) whose bullet costs the copy the fewest points to give up for
    the new one; on a tie the later slot. None when every slot would leave another must-have unnamed."""
    chosen = bullets_of_each_role[role_number - 1]
    points_now = _points_of_the_copy(bullets_of_each_role, requirements)
    cheapest, cheapest_loss = None, None
    for slot in range(BULLETS_NEVER_SWAPPED_OUT, len(chosen)):
        given_up = chosen[slot]
        after_the_swap = _with_a_swap(bullets_of_each_role, role_number, slot, new_bullet)
        if _leaves_a_must_unnamed(given_up, after_the_swap, requirements):
            continue
        loss = points_now - _points_of_the_copy(after_the_swap, requirements)
        if cheapest is None or loss <= cheapest_loss:
            cheapest, cheapest_loss = slot, loss
    return cheapest


def _with_a_swap(bullets_of_each_role: list, role_number: int, slot: int, new_bullet: dict) -> list:
    """A copy of the chosen bullets with `new_bullet` in the role's `slot`; the chosen bullets themselves are untouched."""
    after_the_swap = [list(bullets) for bullets in bullets_of_each_role]
    after_the_swap[role_number - 1][slot] = new_bullet
    return after_the_swap


def _points_of_the_copy(bullets_of_each_role: list, requirements: list) -> int:
    """What the chosen bullets are worth together: every requirement any of them shows, counted once,
    and a measured result's points if any of them states one."""
    shown = set()
    states_a_result = False
    for role_number, bullets in enumerate(bullets_of_each_role, start=1):
        for bullet in bullets:
            shown |= _shown_by(bullet, role_number, requirements)
            if _A_MEASURED_RESULT.search(bullet["text"]):
                states_a_result = True
    return _points_for(shown, requirements) + (WORTH_OF_A_MEASURED_RESULT if states_a_result else 0)


def _leaves_a_must_unnamed(given_up: dict, bullets_of_each_role: list, requirements: list) -> bool:
    """Whether the bullet given up named a must-have that no bullet left in the copy names."""
    for requirement in requirements:
        keyword = requirement["requirement"]
        if (requirement["importance"] == MUST_HAVE and _names_the_requirement(given_up["text"], keyword)
                and not _named_anywhere(bullets_of_each_role, keyword)):
            return True
    return False


def _named_anywhere(bullets_of_each_role: list, keyword: str) -> bool:
    """Whether any chosen bullet of any role names the keyword."""
    for bullets in bullets_of_each_role:
        for bullet in bullets:
            if _names_the_requirement(bullet["text"], keyword):
                return True
    return False
