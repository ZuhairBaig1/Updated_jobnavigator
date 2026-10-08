"""A posting's must-have lines and preferred lines, in the posting's own words.

How it works:
  1. The posting is prepared and cut into lines: saved-as-data forms undone, links and the job board's
     Key Skills labels removed, lines cut at line breaks and bold labels, then cleaned. Nothing is reworded.
  2. Jev says which lines are headings. Code adds the ones the posting itself marks as headings.
  3. Code puts each line under the heading above it.
  4. Jev takes out lines that only sit below a section, such as legal text after a list.
  5. Jev judges each section from its heading and its lines: must-have, preferred, or other.
  6. Jev checks each line of the must-have and preferred sections, with its heading named: keep it,
     move it to the other list, or leave it out. Its answer is acted on only when it is sure.
  7. With a résumé: Jev scores how much of each must-have and preferred line the candidate has actually done.
  8. The match score adds four parts by weight (skills, responsibilities, experience, qualifications),
     each judged by Jev, and scales the total by role fit: whether the candidate does this kind of work.
"""
import asyncio
import html
import logging
import re

from backend.analyzer import decision_scorer, keyword_match
from backend.analyzer.job_requirements import clean_job_text

logger = logging.getLogger("jobnavigator.line_match")


# ── Step 1: cutting the posting into lines ─────────────────────────────────────

BULLET_MARKS = "-•*+●◦▪· \t"
# A line longer than this is a paragraph, and is cut into its sentences.
PARAGRAPH_CHARS = 160
END_OF_SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z])")
MOST_LINES_JUDGED = 200


# The posting's own formatting marks. They are read while cutting, and cleaned off afterwards.
BOLD_RUN = re.compile(r"(\*\*[^*\n]+?\*\*)")
UNDERLINE_ROW = re.compile(r"^[ \t]*[-=]{3,}[ \t]*$")
HASH_HEADING_ROW = re.compile(r"^[ \t]*#{1,6}[ \t]+\S")
BULLET_AT_THE_START = re.compile(r"^[ \t]*(?:[-•+●◦▪·]|\*(?!\*))[ \t]+")
BOLD_RUNS_BACK_TO_BACK = re.compile(r"(?<=[^*\s])\*\*\*\*(?=[^*\s])")
ENDS_A_SENTENCE = re.compile(r"[.!?:)]\s*$")
LONGEST_DECLARED_HEADING_CHARS = 120


WRITTEN_OUT_LINE_BREAK = re.compile(r"\\r\\n|\\n|\\r")
# A character saved as its code, such as \\u00a0, or a broken one such as \\x20;
WRITTEN_OUT_CHARACTER = re.compile(r"\\u([0-9a-fA-F]{4})")
BROKEN_CHARACTER_CODE = re.compile(r"\\x([0-9a-fA-F]{2});?")
# Marks left at the ends of a line: bullets, italics stars, and the backslash that ends a line in some saved text.
MARKS_AT_THE_ENDS_OF_A_LINE = BULLET_MARKS + "\\"
# [Docker](https://site/docker-jobs) keeps its words and loses its address. A bare address is dropped.
LINK_WITH_ITS_WORDS = re.compile(r"\[([^\]\n]*)\]\(\s*(?:https?://|www\.)[^)\s]*\s*\)")
BARE_LINK = re.compile(r"(?:https?://|www\.)[^\s)>\]]+")


def posting_as_plain_text(job_text: str) -> str:
    """The posting before anything reads it: saved-as-data forms undone, and links removed.

    \\n written out as two characters becomes a line break, and codes like &#x20; become their characters.
    Line breaks and formatting marks are kept.
    """
    text = job_text or ""
    was_saved_as_a_quoted_string = "\n" not in text.strip() and WRITTEN_OUT_LINE_BREAK.search(text)
    if was_saved_as_a_quoted_string:
        text = WRITTEN_OUT_LINE_BREAK.sub("\n", text.strip()).strip('"')
        text = WRITTEN_OUT_CHARACTER.sub(lambda code: chr(int(code.group(1), 16)), text)
        text = text.replace('\\"', '"')
    text = html.unescape(text).replace("\xa0", " ")
    text = BROKEN_CHARACTER_CODE.sub(lambda code: chr(int(code.group(1), 16)), text)
    text = _without_the_job_boards_key_skills(_without_links(text))
    return _with_the_job_boards_profile_heading_renamed(text)


def _without_links(text: str) -> str:
    words_only = LINK_WITH_ITS_WORDS.sub(lambda link: f" {link.group(1)} ", text)
    return BARE_LINK.sub("", words_only)


# The job board adds its own "Key Skills" labels at the very end, after its details box. They are search
# labels, mostly repeating the posting, and are left out for now. A "Key Skills" section the employer
# wrote sits higher up and is not touched.
KEY_SKILLS_ROW = re.compile(r"^[\s#*_]*key ?skills[\s*_:]*$", re.IGNORECASE)
NOTE_ABOUT_HIGHLIGHTED_SKILLS = re.compile(r"skills highlighted with.{0,12}are preferred key ?skills", re.IGNORECASE)
ROW_OF_THE_DETAILS_BOX = re.compile(r"^[\s#*_]*(UG|PG|Education|Role Category|Employment Type|Department|Industry Type)\b",
                                    re.IGNORECASE)
ROWS_LOOKED_AT_ABOVE = 8


def _without_the_job_boards_key_skills(text: str) -> str:
    rows = text.split("\n")
    key_skills_rows = [number for number, row in enumerate(rows) if KEY_SKILLS_ROW.match(row)]
    if not key_skills_rows:
        return text
    last = key_skills_rows[-1]
    rows_above = [row for row in rows[:last] if row.strip()][-ROWS_LOOKED_AT_ABOVE:]
    comes_after_the_details_box = any(ROW_OF_THE_DETAILS_BOX.match(row) for row in rows_above)
    has_the_note = bool(NOTE_ABOUT_HIGHLIGHTED_SKILLS.search("\n".join(rows[last:])))
    if comes_after_the_details_box or has_the_note:
        return "\n".join(rows[:last]).rstrip()
    return text


# The job board titles every posting's requirements box "Preferred candidate profile". The employer did not
# choose the word "Preferred", and it makes the lines under it look optional, so the title is made neutral.
JOB_BOARDS_PROFILE_HEADING = re.compile(r"^([\s#*_]*)preferred candidate profile([\s*_:]*)$", re.IGNORECASE | re.MULTILINE)
NEUTRAL_PROFILE_HEADING = "Candidate profile"


def _with_the_job_boards_profile_heading_renamed(text: str) -> str:
    return JOB_BOARDS_PROFILE_HEADING.sub(lambda heading: f"{heading.group(1)}{NEUTRAL_PROFILE_HEADING}{heading.group(2)}", text)


# A page that repeats itself repeats long runs of lines. A run this long that was seen before is dropped.
# A single line that appears twice, such as a tool named in two sections, is kept both times.
LINES_IN_A_REPEATED_BLOCK = 10


def posting_lines(job_text: str) -> list:
    """The posting's own lines, in order. A line's place in this list is its number in the rest of the file."""
    lines = []
    for piece in _pieces_cut_at_line_breaks_and_bold_labels(job_text):
        for line in _split_paragraph(piece["text"]):
            if _is_worth_judging(line):
                lines.append(line)
    return _without_repeated_blocks(lines)[:MOST_LINES_JUDGED]


def _without_repeated_blocks(lines: list) -> list:
    """Drops a run of lines that already appeared, in the same order, earlier in the posting."""
    where_a_block_first_started = {}
    is_part_of_a_repeat = [False] * len(lines)
    for start in range(len(lines) - LINES_IN_A_REPEATED_BLOCK + 1):
        numbers_in_block = range(start, start + LINES_IN_A_REPEATED_BLOCK)
        block = tuple(lines[number].lower() for number in numbers_in_block)
        first_start = where_a_block_first_started.setdefault(block, start)
        does_not_overlap_the_first = start - first_start >= LINES_IN_A_REPEATED_BLOCK
        if does_not_overlap_the_first:
            for number in numbers_in_block:
                is_part_of_a_repeat[number] = True
    return [line for line, repeated in zip(lines, is_part_of_a_repeat) if not repeated]


def headings_the_posting_declares(job_text: str) -> list:
    """Lines the posting itself formats as headings: underlined, or starting with #."""
    return [piece["text"] for piece in _pieces_cut_at_line_breaks_and_bold_labels(job_text)
            if piece["declared_heading"] and len(piece["text"]) <= LONGEST_DECLARED_HEADING_CHARS]


def _pieces_cut_at_line_breaks_and_bold_labels(job_text: str) -> list:
    """The posting cut first, cleaned second.

    Each piece is {"text", "declared_heading", "label_cut_from_it": the bold label that stood before it, or None}.
    """
    pieces = []
    row_above_had_text = False
    for row in posting_as_plain_text(job_text).split("\n"):
        if UNDERLINE_ROW.match(row):
            if pieces and row_above_had_text:
                pieces[-1]["declared_heading"] = True
            row_above_had_text = False
            continue
        starts_with_hashes = bool(HASH_HEADING_ROW.match(row))
        label_before = None
        for raw_piece in _row_cut_at_bold_labels(row):
            text = clean_job_text(raw_piece["text"]).strip(MARKS_AT_THE_ENDS_OF_A_LINE)
            if not text:
                continue
            pieces.append({"text": text, "declared_heading": starts_with_hashes, "label_cut_from_it": label_before})
            label_before = text if raw_piece["is_a_label"] else None
        row_above_had_text = bool(row.strip())
    return pieces


def _labels_cut_from_their_content(job_text: str) -> dict:
    """{bold label: the line that followed it on the same row}."""
    content_after = {}
    for piece in _pieces_cut_at_line_breaks_and_bold_labels(job_text):
        if piece["label_cut_from_it"]:
            content_after[piece["label_cut_from_it"]] = _split_paragraph(piece["text"])[0]
    return content_after


def _row_cut_at_bold_labels(row: str) -> list:
    """A bold label is cut away from the text it is glued to. Bold words inside a sentence stay where they are.

    Returns the row's pieces in order, each {"text", "is_a_label"}.
    """
    is_a_list_item = bool(BULLET_AT_THE_START.match(row))
    row = BOLD_RUNS_BACK_TO_BACK.sub(" ", row)
    parts = [part for part in BOLD_RUN.split(BULLET_AT_THE_START.sub("", row)) if part.strip()]
    if len(parts) < 2:
        return [{"text": row, "is_a_label": False}]

    pieces = []
    for position, part in enumerate(parts):
        part_before = parts[position - 1] if position > 0 else ""
        part_after = parts[position + 1] if position + 1 < len(parts) else ""
        if _is_a_bold_label(part, part_before, part_after, is_a_list_item):
            pieces.append({"text": part, "is_a_label": True})
        elif pieces and not pieces[-1]["is_a_label"]:
            pieces[-1]["text"] += part
        else:
            pieces.append({"text": part, "is_a_label": False})
    return pieces


def _is_bold(part: str) -> bool:
    return bool(BOLD_RUN.fullmatch(part.strip()))


def _is_a_bold_label(part: str, part_before: str, part_after: str, is_a_list_item: bool) -> bool:
    if not _is_bold(part):
        return False
    opens_the_row = not part_before
    closes_the_row = not part_after
    carries_a_colon = part.strip().rstrip("*").rstrip().endswith(":") or part_after.lstrip().startswith(":")
    if opens_the_row and closes_the_row:
        return True
    if opens_the_row:
        return not is_a_list_item and (carries_a_colon or _is_bold(part_after))
    return closes_the_row and bool(ENDS_A_SENTENCE.search(part_before))


def _split_paragraph(line: str) -> list:
    if len(line) <= PARAGRAPH_CHARS:
        return [line]
    return [sentence.strip() for sentence in END_OF_SENTENCE.split(line)]


def _is_worth_judging(line: str) -> bool:
    return len(line) >= 2


# ── What Jev reads, and how it is asked ────────────────────────────────────────

QUESTIONS_PER_CALL = 50
LONGEST_LINE_QUOTED_CHARS = 300


# Jev is asked about every line, so it is shown the whole posting. The longest stored one is 55,619 characters.
POSTING_MAX_CHARS = 100_000


def _posting_state(job_text: str) -> str:
    posting = posting_as_plain_text(job_text)[:POSTING_MAX_CHARS]
    return f"Job posting: {posting}"


def _quoted(line: str) -> str:
    """The line as it appears inside a question."""
    return line[:LONGEST_LINE_QUOTED_CHARS].replace('"', "'")


async def _ask(scorer: str, api_key: str, state: str, questions: dict) -> dict:
    """Sends the questions to Jev, fifty at a time, and returns every answer by its question's name."""
    names = list(questions)
    answers = {}
    calls = []
    for start in range(0, len(names), QUESTIONS_PER_CALL):
        names_in_this_call = names[start:start + QUESTIONS_PER_CALL]
        questions_in_this_call = {name: questions[name] for name in names_in_this_call}
        calls.append(decision_scorer._ask_model(scorer, api_key, state, questions_in_this_call))
    for reply in await asyncio.gather(*calls):
        answers.update(reply.get("answers") or {})
    return answers


# ══ Steps 2 to 4: the headings, and the lines under them ═════════════════════════
#
# Jev reads the whole posting for both of its questions here:
#   2. Jev says which lines are headings. Code adds the ones the posting itself marks.
#   3. Code gives every other line the heading above it. Position is a fact, so nothing is asked.
#   4. Jev says whether each line really belongs to that section, or only happens to sit below it.
#      A line turned away from a section is offered to the heading before it.

HEADING = "heading"
NOT_A_HEADING = "not_a_heading"

HEADING_OR_NOT = {
    HEADING: "A heading or section title, or a sentence that introduces the list after it, such as one ending "
             "'you will bring the following skills:'. It introduces other lines and asks for nothing itself",
    NOT_A_HEADING: "Any other line: a requirement, a duty, a detail, an ordinary sentence or an item in a list",
}
# A line that opens by naming the skills in general and then calls them preferred.
NOTE_THAT_THE_SKILLS_ARE_PREFERRED = re.compile(
    r"^(the )?(following |these |below )?(key ?)?(skills|qualifications)\b[^.,:]*\b(preferred|nice[- ]to[- ]have|"
    r"good[- ]to[- ]have|optional)\b", re.IGNORECASE)

HEADING_QUESTION = 'Is this line of the job posting a heading: "{line}"'

BELONGS = "belongs"
DOES_NOT_BELONG = "does_not_belong"

BELONGS_OR_NOT = {
    BELONGS: "The line is part of that section: it is one of the things the heading introduces",
    DOES_NOT_BELONG: "The line only sits below that heading and is not part of its section: for example a detail "
                     "of the posting, or text about the company, benefits or legal matters that follows a list",
}
BELONGS_QUESTION = (
    'In the job posting, the line "{line}" sits below the heading "{heading}". Is the line part of that section?'
)


async def lines_under_headings(scorer: str, job_text: str) -> dict | None:
    """The posting's headings, and every other line with the heading it belongs under.

    Lines are told apart by their number, so the same words in two sections are two lines.
    Returns {"headings": [texts], "heading_numbers": [their numbers], "sub_heading_numbers": [numbers],
    "lines": [rows]}, where a row is {"number", "line", "heading_number", "heading"} and the heading is
    None for a line that has none.
    Returns None when Jev cannot be asked.
    """
    lines = posting_lines(job_text)
    api_key = decision_scorer._find_api_key(scorer)
    if not lines or not api_key:
        return None

    try:
        heading_numbers = await _numbers_of_the_headings_jev_finds(scorer, api_key, job_text, lines)
        heading_numbers |= _numbers_of_the_headings_code_finds(job_text, lines)
        lines, heading_numbers = _labels_that_are_not_headings_put_back(
            lines, heading_numbers, _labels_cut_from_their_content(job_text))
        rows = _each_line_with_the_heading_above_it(lines, heading_numbers)
        await _settle_which_section_each_line_belongs_to(scorer, api_key, job_text, rows, lines, heading_numbers)
    except Exception as error:
        logger.warning(f"The posting's headings could not be read ({type(error).__name__}: {error})")
        return None

    numbers_in_order = sorted(heading_numbers)
    sub_heading_texts = _texts_of_sub_headings(job_text)
    return {"headings": [lines[number] for number in numbers_in_order], "heading_numbers": numbers_in_order,
            "sub_heading_numbers": [number for number in numbers_in_order if lines[number] in sub_heading_texts],
            "lines": rows}


# Step 2: which lines are headings.

async def _numbers_of_the_headings_jev_finds(scorer: str, api_key: str, job_text: str, lines: list) -> set:
    questions = {}
    for number, line in enumerate(lines):
        questions[f"h{number}"] = {
            "type": "choice",
            "instructions": HEADING_QUESTION.format(line=_quoted(line)),
            "criteria": HEADING_OR_NOT,
        }
    answers = await _ask(scorer, api_key, _posting_state(job_text), questions)
    return {number for number in range(len(lines)) if answers[f"h{number}"]["choice"] == HEADING}


def _numbers_of_the_headings_code_finds(job_text: str, lines: list) -> set:
    """Headings the posting declares, sentences that introduce a list, and notes that the skills are preferred."""
    heading_texts = set(headings_the_posting_declares(job_text)) | set(_lines_that_introduce_a_list(job_text))
    heading_texts |= set(_notes_that_the_skills_are_preferred(lines))
    return {number for number, line in enumerate(lines) if line in heading_texts}


def _labels_that_are_not_headings_put_back(lines: list, heading_numbers: set, content_after: dict) -> tuple:
    """A bold label was cut off in case it is a heading. If it is not one, it rejoins the line after it.

    Joining two lines moves the lines below them up by one, so the headings' new numbers are returned too.
    """
    new_lines = []
    new_heading_numbers = set()
    number = 0
    while number < len(lines):
        line = lines[number]
        if _is_a_label_to_put_back(number, lines, heading_numbers, content_after):
            new_lines.append(f"{line} {lines[number + 1]}")
            number += 2
            continue
        if number in heading_numbers:
            new_heading_numbers.add(len(new_lines))
        new_lines.append(line)
        number += 1
    return new_lines, new_heading_numbers


def _is_a_label_to_put_back(number: int, lines: list, heading_numbers: set, content_after: dict) -> bool:
    next_number = number + 1
    if number in heading_numbers or next_number >= len(lines) or next_number in heading_numbers:
        return False
    return content_after.get(lines[number]) == lines[next_number]


def _lines_that_introduce_a_list(job_text: str) -> list:
    """Lines that end with a colon and are followed by a list: they are headings whatever their length."""
    rows = [row for row in clean_job_text(posting_as_plain_text(job_text)).split("\n") if row.strip()]
    lead_ins = []
    for row, next_row in zip(rows, rows[1:]):
        ends_with_a_colon = row.rstrip().endswith(":")
        if ends_with_a_colon and _is_a_list_item(next_row) and not _is_a_list_item(row):
            lead_ins.append(row.strip(BULLET_MARKS))
    return lead_ins


def _texts_of_sub_headings(job_text: str) -> set:
    """List items that introduce a list of their own: they end with a colon and the next item is indented deeper.

    "- Working knowledge of:" followed by "    - DynamoDB" is one. It sits inside a section and is not a new one.
    """
    rows = [row for row in posting_as_plain_text(job_text).split("\n") if row.strip()]
    sub_headings = set()
    for row, next_row in zip(rows, rows[1:]):
        both_are_list_items = BULLET_AT_THE_START.match(row) and BULLET_AT_THE_START.match(next_row)
        text = clean_job_text(row).strip(BULLET_MARKS)
        if both_are_list_items and _indent_of(next_row) > _indent_of(row) and text.endswith(":"):
            sub_headings.add(text)
    return sub_headings


def _indent_of(row: str) -> int:
    return len(row) - len(row.lstrip(" \t"))


def _notes_that_the_skills_are_preferred(lines: list) -> list:
    """Lines such as "Skills highlighted are preferred": they speak for the lines after them, so they are headings."""
    return [line for line in lines if NOTE_THAT_THE_SKILLS_ARE_PREFERRED.search(line)]


def _is_a_list_item(row: str) -> bool:
    """A row that starts with a bullet mark or an indent."""
    return row[:1] in BULLET_MARKS


# Step 3: the heading above each line, by position.

def _each_line_with_the_heading_above_it(lines: list, heading_numbers: set) -> list:
    """A row for every line that is not a heading, with the last heading that comes before it."""
    rows = []
    heading_above = None
    for number, line in enumerate(lines):
        if number in heading_numbers:
            heading_above = number
        else:
            row = {"number": number, "line": line}
            _give_the_row_this_heading(row, heading_above, lines)
            rows.append(row)
    return rows


def _give_the_row_this_heading(row: dict, heading_number, lines: list) -> None:
    row["heading_number"] = heading_number
    row["heading"] = lines[heading_number] if heading_number is not None else None


# Step 4: whether each line belongs to the section it sits in.

async def _settle_which_section_each_line_belongs_to(scorer: str, api_key: str, job_text: str, rows: list,
                                                     lines: list, heading_numbers: set) -> None:
    """A line that does not belong under its heading is offered to the heading before that one.

    This is how a line that follows a sub-list goes back to the section the sub-list sits in.
    A line that belongs to neither is left without a heading, such as legal text after a list.
    """
    rows_with_a_heading = [row for row in rows if row["heading_number"] is not None]
    rows_turned_away = await _rows_that_do_not_belong(scorer, api_key, job_text, rows_with_a_heading)
    for row in rows_turned_away:
        _give_the_row_this_heading(row, _heading_before(row["heading_number"], heading_numbers), lines)

    rows_offered_again = [row for row in rows_turned_away if row["heading_number"] is not None]
    for row in await _rows_that_do_not_belong(scorer, api_key, job_text, rows_offered_again):
        _give_the_row_this_heading(row, None, lines)


async def _rows_that_do_not_belong(scorer: str, api_key: str, job_text: str, rows_to_check: list) -> list:
    """The rows Jev says only sit below their heading and are not part of its section."""
    if not rows_to_check:
        return []
    questions = {}
    for number, row in enumerate(rows_to_check):
        questions[f"b{number}"] = {
            "type": "choice",
            "instructions": BELONGS_QUESTION.format(line=_quoted(row["line"]), heading=_quoted(row["heading"])),
            "criteria": BELONGS_OR_NOT,
        }
    answers = await _ask(scorer, api_key, _posting_state(job_text), questions)
    return [row for number, row in enumerate(rows_to_check) if answers[f"b{number}"]["choice"] == DOES_NOT_BELONG]


def _heading_before(heading_number: int, heading_numbers: set):
    """The number of the nearest heading above this one, or None when it is the first."""
    earlier_headings = [number for number in heading_numbers if number < heading_number]
    return max(earlier_headings) if earlier_headings else None


# ══ Steps 5 and 6: the must-haves and the preferred ════════════════════════════
#
# 5. Each section is judged from its heading and the lines under it: must-have, preferred, or other.
#    A sub-heading inside a section takes that section's judgement.
#    A heading that calls its skills preferred is preferred. Sections judged "other" are left out.
# 6. Each line of the kept sections is then judged with its heading named: keep it where it is, move it
#    to the other list, or leave it out. Jev's answer is acted on only when it is sure: 90% to move
#    a line between the lists, 60% to leave it out.
#    A line labelled "Secondary:" or "Preferred:" is preferred whatever Jev answers.
# Where the posting has no such sections, both lists are empty and the result says nothing was found.
# Last, a labelled years-of-experience line found anywhere in the posting is added to the must-haves.

REQUIREMENTS_SECTION = "requirements"
PREFERRED_SECTION = "preferred"
ANOTHER_SECTION = "other"

# Step 5: what each section holds.
WHAT_A_SECTION_HOLDS = {
    REQUIREMENTS_SECTION: "The lines under it are what the candidate must have for this job: skills, tools, "
                          "experience, years, qualifications or education",
    PREFERRED_SECTION: "The lines under it are extras the posting would like but does not require",
    ANOTHER_SECTION: "The lines under it are something else: duties, a summary, details of the posting, "
                     "the company, benefits",
}
SECTION_QUESTION = (
    'In the job posting, the heading "{heading}" has these lines under it: {lines}. What do those lines hold? '
    "Judge by the lines themselves and by where the section sits in the posting, not by the words of the heading. "
    "A heading that contains the word preferred does not make its section preferred.")
MOST_LINES_OF_A_SECTION_SHOWN = 8
LONGEST_SECTION_LINE_SHOWN_CHARS = 120
NO_LINES_UNDER_THE_HEADING = "(none)"

# Step 6: where each line of a kept section belongs.
IN_MUST_HAVE = "must_have"
IN_PREFERRED = "preferred"
IN_NEITHER = "other"
WHERE_A_LINE_BELONGS = {
    IN_MUST_HAVE: "Must-have: the line is something the candidate must have for this job",
    IN_PREFERRED: "Preferred: the line is something the posting would like but does not require",
    IN_NEITHER: "Other: the line is not about what the candidate should have. It is what the person will do in "
                "the job or a description of the role, or a detail such as the location, the pay, the notice "
                "period, the shift or working hours, how the interview is held, or text about the company",
}
QUESTION_FOR_A_MUST_HAVE_LINE = (
    'In the job posting, the line "{line}" is under the heading "{heading}", which is part of the must-haves. '
    "Based on the job posting, would you keep this line in the must-haves, move it to preferred, or is it other?")
QUESTION_FOR_A_PREFERRED_LINE = (
    'In the job posting, the line "{line}" is under the heading "{heading}", which is part of the preferred. '
    "Based on the job posting, does this line belong in preferred, in the must-haves, or is it other?")
# A line moves between must-have and preferred only when Jev is this sure.
SURE_ENOUGH_TO_MOVE_A_LINE = 0.90
# A line is left out as "other" at a lower bar: a line about the shift or the location is seldom a clear case.
SURE_ENOUGH_TO_LEAVE_A_LINE_OUT = 0.60
# The job board's education lines, "UG: ..." and "PG: ...", are never left out, unless they say no degree is needed.
EDUCATION_LINE_OF_THE_JOB_BOARD = re.compile(r"^(UG|PG)\s*:", re.IGNORECASE)
SAYS_NO_DEGREE_IS_NEEDED = re.compile(r"\bnot required\b|\bnot needed\b|\bno degree\b", re.IGNORECASE)
# A line that opens with a label such as "Secondary:" or "Preferred:" is preferred, whatever Jev answers.
LABELLED_AS_PREFERRED = re.compile(
    r"^(secondary|preferred|optional|desirable|good[- ]to[- ]have|nice[- ]to[- ]have)"
    r"(\s+(skills?|skill ?sets?|qualifications?|requirements?|experience))?\s*[:\-–]", re.IGNORECASE)

# Words by which a line calls itself optional. "(Azure preferred)" is one option inside a line, so
# words in brackets are not counted.
SAYS_IT_IS_PREFERRED = re.compile(
    r"\b(preferred|nice[- ]to[- ]have|good[- ]to[- ]have|a plus|is a plus|bonus|an advantage|desirable|optional)\b",
    re.IGNORECASE)
TEXT_IN_BRACKETS = re.compile(r"\([^)]*\)")
SAYS_IT_IS_REQUIRED = re.compile(r"\b(required|requirements?|must[- ]have|mandatory|essential)\b", re.IGNORECASE)
# "Preferred candidate profile" describes the person wanted, not skills that are optional.
IS_ABOUT_THE_CANDIDATE = re.compile(r"\b(candidates?|profile|applicants?)\b", re.IGNORECASE)

# A detail line that states the years of experience, such as "Experience: 10-15 Years" or "Exp: 6 to 12".
# Job boards often put it in the header, away from the skills, so it is looked for across the whole posting.
LABELLED_YEARS_OF_EXPERIENCE = re.compile(
    r"^(total |overall |relevant |work |required )?(experience|exp)\.?( level| range| required| needed)?"
    r"\s*[:\-–]\s*\D{0,15}\d", re.IGNORECASE)
LONGEST_YEARS_LINE_CHARS = 90

FROM_HEADINGS = "headings"
# The posting has no section that holds must-haves or preferred skills, so nothing could be read from it.
NOTHING_FOUND = "nothing"


async def must_haves_and_preferred(scorer: str, job_text: str) -> dict | None:
    """The posting's must-have lines and preferred lines, in the posting's own words.

    Returns {"must_have": [...], "preferred": [...], "found_from": "headings" or "nothing"},
    or None when Jev cannot be asked.
    """
    sorted_lines = await lines_of_the_posting_by_what_they_are(scorer, job_text)
    if sorted_lines is None:
        return None
    return {"must_have": sorted_lines["must_have"], "preferred": sorted_lines["preferred"],
            "found_from": sorted_lines["found_from"]}


async def lines_of_the_posting_by_what_they_are(scorer: str, job_text: str) -> dict | None:
    """The posting's must-have lines, preferred lines and duties, in the posting's own words.

    Returns {"must_have": [...], "preferred": [...], "duties": [...], "found_from": "headings" or "nothing"},
    or None when Jev cannot be asked.
    """
    under_headings = await lines_under_headings(scorer, job_text)
    if under_headings is None:
        return None

    api_key = decision_scorer._find_api_key(scorer)
    try:
        section_of = await _what_each_section_holds(scorer, api_key, job_text, under_headings)
        _sub_headings_take_the_section_they_sit_in(section_of, under_headings)
        _headings_that_say_preferred_are_preferred(section_of, under_headings)
        lines_of_kept_sections = _lines_of_must_have_and_preferred_sections(under_headings, section_of)
        places, duties = await asyncio.gather(
            _where_each_line_belongs(scorer, api_key, job_text, lines_of_kept_sections),
            _lines_of_the_duties_sections(scorer, api_key, job_text, under_headings, section_of))
        places = _lines_labelled_as_preferred_are_preferred(lines_of_kept_sections, places)
        places = _education_lines_are_not_left_out(lines_of_kept_sections, places)
    except Exception as error:
        logger.warning(f"The posting's sections could not be judged ({type(error).__name__}: {error})")
        return None

    sorted_lines = {"must_have": [], "preferred": []}
    for row, place in zip(lines_of_kept_sections, places):
        if place != IN_NEITHER:
            sorted_lines[place].append(row["line"])
    found_any = sorted_lines["must_have"] or sorted_lines["preferred"]
    sorted_lines["found_from"] = FROM_HEADINGS if found_any else NOTHING_FOUND
    _add_the_years_of_experience_stated_anywhere(sorted_lines, job_text)
    sorted_lines["duties"] = duties
    return sorted_lines


# The sections that hold neither must-haves nor preferred skills are asked one more thing: are they the duties?
ARE_DUTIES = "duties"
ARE_NOT_DUTIES = "not_duties"
DUTIES_OR_NOT = {
    ARE_DUTIES: "Duties: the lines say what the person will do in the job",
    ARE_NOT_DUTIES: "Something else: a summary of the role, details of the posting, the company, its culture "
                    "or its benefits",
}
DUTIES_QUESTION = (
    'In the job posting, the heading "{heading}" has these lines under it: {lines}. '
    "Are those lines the duties of the job, the things the person will do in the role?")


async def _lines_of_the_duties_sections(scorer: str, api_key: str, job_text: str, under_headings: dict,
                                        section_of: dict) -> list:
    """The lines of every section that is neither must-have nor preferred and that Jev says holds duties."""
    lines_under = {}
    for row in under_headings["lines"]:
        if section_of.get(row["heading_number"]) == ANOTHER_SECTION:
            lines_under.setdefault(row["heading_number"], []).append(row["line"])
    if not lines_under:
        return []

    heading_text = dict(zip(under_headings["heading_numbers"], under_headings["headings"]))
    questions = {}
    for heading_number in lines_under:
        questions[f"d{heading_number}"] = {
            "type": "choice",
            "instructions": DUTIES_QUESTION.format(heading=_quoted(heading_text[heading_number]),
                                                   lines=_lines_under_as_text(under_headings, heading_number)),
            "criteria": DUTIES_OR_NOT,
        }
    answers = await _ask(scorer, api_key, _posting_state(job_text), questions)

    duties = []
    for heading_number, lines in lines_under.items():
        if answers[f"d{heading_number}"]["choice"] == ARE_DUTIES:
            duties += lines
    return duties


def _add_the_years_of_experience_stated_anywhere(sorted_lines: dict, job_text: str) -> None:
    """A labelled years-of-experience line is a must-have wherever it sits; it goes first when it was missing."""
    already_listed = sorted_lines["must_have"] + sorted_lines["preferred"]
    missing_years_lines = []
    for line in posting_lines(job_text):
        if _states_the_years_of_experience(line) and line not in already_listed + missing_years_lines:
            missing_years_lines.append(line)
    sorted_lines["must_have"] = missing_years_lines + sorted_lines["must_have"]


def _states_the_years_of_experience(line: str) -> bool:
    return len(line) <= LONGEST_YEARS_LINE_CHARS and bool(LABELLED_YEARS_OF_EXPERIENCE.match(line))


# Step 5: what each section holds.

async def _what_each_section_holds(scorer: str, api_key: str, job_text: str, under_headings: dict) -> dict:
    """{heading's number: "requirements", "preferred" or "other"}, judged from each heading and its lines."""
    numbers_and_headings = list(zip(under_headings["heading_numbers"], under_headings["headings"]))
    if not numbers_and_headings:
        return {}
    questions = {}
    for heading_number, heading in numbers_and_headings:
        questions[f"s{heading_number}"] = {
            "type": "choice",
            "instructions": SECTION_QUESTION.format(heading=_quoted(heading),
                                                    lines=_lines_under_as_text(under_headings, heading_number)),
            "criteria": WHAT_A_SECTION_HOLDS,
        }
    answers = await _ask(scorer, api_key, _posting_state(job_text), questions)
    return {heading_number: answers[f"s{heading_number}"]["choice"] for heading_number, _ in numbers_and_headings}


def _lines_under_as_text(under_headings: dict, heading_number: int) -> str:
    """The first few lines under a heading, as they are written into the question."""
    lines_under = [row["line"] for row in under_headings["lines"] if row["heading_number"] == heading_number]
    shown = [_quoted(line)[:LONGEST_SECTION_LINE_SHOWN_CHARS] for line in lines_under[:MOST_LINES_OF_A_SECTION_SHOWN]]
    return " / ".join(shown) or NO_LINES_UNDER_THE_HEADING


def _lines_of_must_have_and_preferred_sections(under_headings: dict, section_of: dict) -> list:
    """Every line of a kept section, as {"line", "heading", "list": the list its heading puts it in}."""
    list_of_section = {REQUIREMENTS_SECTION: IN_MUST_HAVE, PREFERRED_SECTION: IN_PREFERRED}
    kept_lines = []
    for row in under_headings["lines"]:
        section = section_of.get(row["heading_number"])
        if section in list_of_section:
            kept_lines.append({"line": row["line"], "heading": row["heading"], "list": list_of_section[section]})
    return kept_lines


# Step 6: where each line of a kept section belongs.

async def _where_each_line_belongs(scorer: str, api_key: str, job_text: str, lines_of_kept_sections: list) -> list:
    """For every line, in order: "must_have", "preferred" or "other". A line moves only when Jev is sure."""
    if not lines_of_kept_sections:
        return []
    questions = {}
    for number, row in enumerate(lines_of_kept_sections):
        question = QUESTION_FOR_A_MUST_HAVE_LINE if row["list"] == IN_MUST_HAVE else QUESTION_FOR_A_PREFERRED_LINE
        questions[f"w{number}"] = {
            "type": "choice",
            "instructions": question.format(line=_quoted(row["line"]), heading=_quoted(row["heading"])),
            "criteria": WHERE_A_LINE_BELONGS,
        }
    answers = await _ask(scorer, api_key, _posting_state(job_text), questions)

    places = []
    for number, row in enumerate(lines_of_kept_sections):
        places.append(_place_jev_is_sure_of(answers[f"w{number}"], place_from_the_heading=row["list"]))
    return places


def _lines_labelled_as_preferred_are_preferred(lines_of_kept_sections: list, places: list) -> list:
    """A line such as "Secondary: Terraform" states itself as preferred, so it goes to the preferred list."""
    corrected_places = []
    for row, place in zip(lines_of_kept_sections, places):
        if LABELLED_AS_PREFERRED.match(row["line"]):
            corrected_places.append(IN_PREFERRED)
        else:
            corrected_places.append(place)
    return corrected_places


def _education_lines_are_not_left_out(lines_of_kept_sections: list, places: list) -> list:
    """"UG: Any Graduate" stays in the list its heading put it in, even when Jev would set it aside."""
    corrected_places = []
    for row, place in zip(lines_of_kept_sections, places):
        is_an_education_line = bool(EDUCATION_LINE_OF_THE_JOB_BOARD.match(row["line"]))
        asks_for_a_degree = not SAYS_NO_DEGREE_IS_NEEDED.search(row["line"])
        if place == IN_NEITHER and is_an_education_line and asks_for_a_degree:
            corrected_places.append(row["list"])
        else:
            corrected_places.append(place)
    return corrected_places


def _place_jev_is_sure_of(answer: dict, place_from_the_heading: str) -> str:
    """The place Jev chose when it is sure enough; otherwise the place the heading gave the line."""
    likelihoods = answer.get("probabilities") or {answer["choice"]: 1.0}
    for place, likelihood in likelihoods.items():
        bar = SURE_ENOUGH_TO_LEAVE_A_LINE_OUT if place == IN_NEITHER else SURE_ENOUGH_TO_MOVE_A_LINE
        if place != place_from_the_heading and float(likelihood) >= bar:
            return place
    return place_from_the_heading


def _sub_headings_take_the_section_they_sit_in(section_of: dict, under_headings: dict) -> None:
    """A sub-heading's list is part of the section around it, so it gets that section's judgement, not its own."""
    section_around = None
    for heading_number in under_headings["heading_numbers"]:
        is_a_sub_heading = heading_number in under_headings["sub_heading_numbers"]
        if is_a_sub_heading and section_around is not None:
            section_of[heading_number] = section_around
        else:
            section_around = section_of[heading_number]


def _headings_that_say_preferred_are_preferred(section_of: dict, under_headings: dict) -> None:
    """A heading that states its skills as preferred is preferred whatever Jev answered."""
    for heading_number, heading in zip(under_headings["heading_numbers"], under_headings["headings"]):
        if _states_its_skills_as_preferred(heading):
            section_of[heading_number] = PREFERRED_SECTION


def _states_its_skills_as_preferred(heading: str) -> bool:
    if SAYS_IT_IS_REQUIRED.search(heading) or IS_ABOUT_THE_CANDIDATE.search(heading):
        return False
    return _calls_itself_preferred(heading)


def _calls_itself_preferred(line: str) -> bool:
    outside_brackets = TEXT_IN_BRACKETS.sub("", line)
    return bool(SAYS_IT_IS_PREFERRED.search(outside_brackets))


# ══ Step 7: how much of each line the résumé shows ══════════════════════════════
#
# Jev reads the résumé and the posting, and scores each must-have and preferred line on its own.
# A score, not a yes or no: a line that names five tools is partly covered by a résumé that shows three.
#
# In the averages each named tool counts once, however many lines name it. A line's weight is the sum of
# its tools' shares: a tool named in three lines gives each of them a third. A line that names no tool,
# such as years of experience or a degree, counts as one. A tool that is a must-have is not counted again
# in the preferred average.

# The levels Jev scores a line on, lowest first, with the share of the line each one stands for.
COVERAGE_LEVELS = [
    ("None of the requirement is demonstrated", 0.0),
    ("A small portion of the requirement is demonstrated", 0.25),
    ("About half of the requirement is demonstrated", 0.50),
    ("Most of the requirement is demonstrated", 0.75),
    ("The entire requirement is demonstrated", 1.0),
]
COVERAGE_QUESTION = (
    'To what extent does the candidate\'s resume demonstrate the requirement: "{line}"? '
    "Judge the requirement as a whole based only on evidence in the resume. "
    "Consider actual work or project experience stronger evidence than a skill that is merely listed. "
    "For requirements containing multiple components, consider the overall proportion that is demonstrated. "
    "Where the requirement provides alternatives, satisfying any stated alternative counts as satisfying that part."
)


def _resume_and_posting_state(resume_text: str, job_text: str) -> str:
    resume = (resume_text or "")[:decision_scorer.RESUME_MAX_CHARS]
    return f"Resume: {resume}\n\n{_posting_state(job_text)}"


async def resume_against_the_posting(scorer: str, resume_text: str, job_text: str) -> dict | None:
    """How much of each must-have and preferred line the résumé shows, from 0 (none) to 1 (all of it).

    Returns {"must_have": [{"line", "covered", "weight"}], "preferred": [{"line", "covered", "weight"}],
    "must_have_covered": average or None, "preferred_covered": average or None, "found_from": ...},
    or None when Jev cannot be asked.
    """
    lists = await must_haves_and_preferred(scorer, job_text)
    if lists is None:
        return None

    api_key = decision_scorer._find_api_key(scorer)
    lines = lists["must_have"] + lists["preferred"]
    try:
        covered = await _how_much_of_each_line_is_covered(scorer, api_key, resume_text, job_text, lines)
    except Exception as error:
        logger.warning(f"The resume could not be scored against the posting ({type(error).__name__}: {error})")
        return None

    must_have_weights = _weights_so_each_tool_counts_once(lists["must_have"])
    preferred_weights = _weights_so_each_tool_counts_once(
        lists["preferred"], tools_counted_already=_tools_named_in_any_of(lists["must_have"]))
    must_have = _lines_with_their_coverage(lists["must_have"], covered[:len(lists["must_have"])], must_have_weights)
    preferred = _lines_with_their_coverage(lists["preferred"], covered[len(lists["must_have"]):], preferred_weights)
    return {"must_have": must_have, "preferred": preferred,
            "must_have_covered": _average_coverage(must_have), "preferred_covered": _average_coverage(preferred),
            "found_from": lists["found_from"]}


async def _how_much_of_each_line_is_covered(scorer: str, api_key: str, resume_text: str, job_text: str,
                                            lines: list) -> list:
    """For every line, in order: the share of it the résumé shows, from 0 to 1."""
    if not lines:
        return []
    questions = {}
    for number, line in enumerate(lines):
        questions[f"c{number}"] = {
            "type": "score",
            "instructions": COVERAGE_QUESTION.format(line=_quoted(line)),
            "criteria": [description for description, _share in COVERAGE_LEVELS],
        }
    answers = await _ask(scorer, api_key, _resume_and_posting_state(resume_text, job_text), questions)
    return [decision_scorer._share_earned(answers[f"c{number}"], COVERAGE_LEVELS) for number in range(len(lines))]


def _tools_named_in(line: str) -> set:
    """The named tools in a line, by their usual names, going by the project's skills list."""
    skills = keyword_match.skills_in(line)
    return {name for name, found in skills.items() if found["type"] == keyword_match.HARD_SKILL}


def _tools_named_in_any_of(lines: list) -> set:
    tools = set()
    for line in lines:
        tools |= _tools_named_in(line)
    return tools


def _weights_so_each_tool_counts_once(lines: list, tools_counted_already: set = frozenset()) -> list:
    """For every line, in order: how much it counts in the average."""
    tools_of_each_line = [_tools_named_in(line) for line in lines]
    lines_naming = {}
    for tools in tools_of_each_line:
        for tool in tools:
            lines_naming[tool] = lines_naming.get(tool, 0) + 1

    weights = []
    for tools in tools_of_each_line:
        if not tools:
            weights.append(1.0)
            continue
        tools_not_counted_yet = tools - tools_counted_already
        weights.append(sum(1 / lines_naming[tool] for tool in tools_not_counted_yet))
    return weights


def _lines_with_their_coverage(lines: list, covered: list, weights: list) -> list:
    return [{"line": line, "covered": round(share, 2), "weight": round(weight, 2)}
            for line, share, weight in zip(lines, covered, weights)]


def _average_coverage(lines_with_coverage: list):
    """The share covered across the lines, each counted by its weight; None when nothing counts."""
    total_weight = sum(row["weight"] for row in lines_with_coverage)
    if not total_weight:
        return None
    return round(sum(row["covered"] * row["weight"] for row in lines_with_coverage) / total_weight, 2)



# ── The ATS score ─────────────────────────────────────────────────────────────
#
# Jev judges three things: how much of each requirement the résumé covers, how important each requirement is
# to the job, and whether the candidate's career is in the occupation the posting is for. The rest is arithmetic:
#     W = how much a line counts, from 1 (a formality) to 4 (what the job is mainly about)
#     M = the coverage of the must-have lines, averaged by W
#     P = the coverage of the preferred lines, averaged by W
#     coverage = 0.8 x M + 0.2 x P
#     ATS = 100 x coverage x the share kept for the occupation
# A posting with no preferred lines is scored on its must-haves alone, and the other way round.
#
# Without the occupation step, a data engineer who has used the same tools scores close to a DevOps engineer.

SHARE_OF_MUST_HAVES_IN_THE_ATS_SCORE = 0.8
SHARE_OF_PREFERRED_IN_THE_ATS_SCORE = 0.2

# How much a requirement line counts, judged from the posting alone. A degree counts less than the job's main skill.
IMPORTANCE_LEVELS = [
    ("A formality or a general trait, such as a degree, a soft skill or a way of working", 1.0),
    ("A supporting skill: useful in the job, but not what the job is mainly about", 2.0),
    ("An important skill for the job's main work", 3.0),
    ("Central: the job is mainly about this", 4.0),
]
IMPORTANCE_QUESTION = (
    'The job posting asks for this: "{line}". How important is it to this job, going by the whole posting?')

# Each answer with the share of the coverage score the résumé keeps.
# The two top levels keep everything and the next keeps most: what such a candidate lacks has already
# cost them in the coverage of the lines, and should not be charged a second time here.
# The levels follow the usual scale for how well past work transfers to a job: by how much would have to be learned.
OCCUPATION_LEVELS = [
    ("A different occupation: almost all of the job's main work would be new to the candidate", 0.0),
    ("A different occupation with a few skills in common: most of the main work would be new", 0.2),
    ("A neighbouring occupation: the candidate did part of this work in support of a different main job, "
     "and the essential duties would have to be learned", 0.4),
    ("A closely related occupation: the candidate did much of this work, and some of the main duties would "
     "have to be learned", 0.8),
    ("The same occupation with a different emphasis: only a few duties would be new", 1.0),
    ("The same occupation: these are the roles the candidate has held, and little or nothing would be new", 1.0),
]
OCCUPATION_QUESTION = (
    "Would a recruiter filling this job see the candidate's career so far as a career in this same occupation? "
    "Judge by the job titles and the main work of each role in the resume, not by which tools are named. "
    "Think of how much of the job's main work the candidate would have to learn from scratch.")


async def ats_score(scorer: str, resume_text: str, job_text: str) -> dict | None:
    """The ATS score out of 100, with the coverage and importance of each must-have and preferred line behind it.

    Returns {"ats_score", "coverage_score", "same_occupation", "must_have_coverage", "preferred_coverage",
    "must_have": [rows], "preferred": [rows], "found_from"}. A row is {"line", "covered", "importance", ...}.
    The coverage score is the score before the occupation step. Both scores are None when the posting gave
    no lines. Returns None when Jev cannot be asked.
    """
    coverage = await resume_against_the_posting(scorer, resume_text, job_text)
    if coverage is None:
        return None
    rows = coverage["must_have"] + coverage["preferred"]
    try:
        importance, same_occupation = await asyncio.gather(
            _how_important_each_line_is(scorer, job_text, [row["line"] for row in rows]),
            _share_kept_for_the_occupation(scorer, resume_text, job_text))
    except Exception as error:
        logger.warning(f"The importance or the occupation could not be judged ({type(error).__name__}: {error})")
        return None
    for row, how_important in zip(rows, importance):
        row["importance"] = round(how_important, 2)

    must_have_coverage = _coverage_averaged_by_importance(coverage["must_have"])
    preferred_coverage = _coverage_averaged_by_importance(coverage["preferred"])
    coverage_score = _coverage_score_out_of_100(must_have_coverage, preferred_coverage)
    ats = None if coverage_score is None else round(coverage_score * same_occupation)
    return {"ats_score": ats, "coverage_score": coverage_score, "same_occupation": round(same_occupation, 2),
            "must_have_coverage": must_have_coverage, "preferred_coverage": preferred_coverage,
            "must_have": coverage["must_have"], "preferred": coverage["preferred"],
            "found_from": coverage["found_from"]}


async def _how_important_each_line_is(scorer: str, job_text: str, lines: list) -> list:
    """For every line, in order: how much it counts, from 1 to 4. Judged from the posting, never the résumé."""
    if not lines:
        return []
    api_key = decision_scorer._find_api_key(scorer)
    questions = {}
    for number, line in enumerate(lines):
        questions[f"i{number}"] = {
            "type": "score",
            "instructions": IMPORTANCE_QUESTION.format(line=_quoted(line)),
            "criteria": [description for description, _counts_for in IMPORTANCE_LEVELS],
        }
    answers = await _ask(scorer, api_key, _posting_state(job_text), questions)
    return [decision_scorer._share_earned(answers[f"i{number}"], IMPORTANCE_LEVELS) for number in range(len(lines))]


def _coverage_averaged_by_importance(rows: list):
    """The coverage of a list of lines, a more important line counting for more; None for an empty list."""
    total_importance = sum(row["importance"] for row in rows)
    if not total_importance:
        return None
    return sum(row["covered"] * row["importance"] for row in rows) / total_importance


async def _share_kept_for_the_occupation(scorer: str, resume_text: str, job_text: str) -> float:
    """From 0 to 1: how far the candidate's career is in the occupation the posting is for."""
    api_key = decision_scorer._find_api_key(scorer)
    question = {"type": "score", "instructions": OCCUPATION_QUESTION,
                "criteria": [description for description, _share in OCCUPATION_LEVELS]}
    answers = await _ask(scorer, api_key, _resume_and_posting_state(resume_text, job_text), {"occupation": question})
    return decision_scorer._share_earned(answers["occupation"], OCCUPATION_LEVELS)


def _coverage_score_out_of_100(must_have_coverage, preferred_coverage):
    if must_have_coverage is None and preferred_coverage is None:
        return None
    if preferred_coverage is None:
        return round(100 * must_have_coverage)
    if must_have_coverage is None:
        return round(100 * preferred_coverage)
    return round(100 * (SHARE_OF_MUST_HAVES_IN_THE_ATS_SCORE * must_have_coverage
                        + SHARE_OF_PREFERRED_IN_THE_ATS_SCORE * preferred_coverage))


# The same question for a duty: has the candidate done this kind of work, whatever words the résumé uses?
DUTY_QUESTION = (
    'The job involves this duty: "{line}". How much of it has the candidate actually done, going by the resume? '
    "Judge the work the candidate did, not the words used; the same work described differently counts.")

# What kind of requirement a must-have or preferred line is. Each kind is scored as its own part.
A_SKILL = "skill"
EXPERIENCE = "experience"
A_QUALIFICATION = "qualification"
KINDS_OF_REQUIREMENT = {
    A_SKILL: "A skill, tool, technology, ability or area of knowledge",
    EXPERIENCE: "A length of experience or a level of seniority, such as a number of years",
    A_QUALIFICATION: "A degree, a level of education, a certification or a licence",
}
KIND_QUESTION = 'What kind of requirement is this line of the job posting: "{line}"'


async def _how_much_of_each_duty_is_done(scorer: str, api_key: str, resume_text: str, job_text: str,
                                         duties: list) -> list:
    """For every duty, in order: the share of it the résumé shows, from 0 to 1."""
    if not duties:
        return []
    questions = {}
    for number, line in enumerate(duties):
        questions[f"u{number}"] = {
            "type": "score",
            "instructions": DUTY_QUESTION.format(line=_quoted(line)),
            "criteria": [description for description, _share in COVERAGE_LEVELS],
        }
    answers = await _ask(scorer, api_key, _resume_and_posting_state(resume_text, job_text), questions)
    return [decision_scorer._share_earned(answers[f"u{number}"], COVERAGE_LEVELS) for number in range(len(duties))]


async def _kind_of_each_line(scorer: str, api_key: str, job_text: str, lines: list) -> list:
    """For every line, in order: "skill", "experience" or "qualification"."""
    if not lines:
        return []
    questions = {}
    for number, line in enumerate(lines):
        questions[f"k{number}"] = {
            "type": "choice",
            "instructions": KIND_QUESTION.format(line=_quoted(line)),
            "criteria": KINDS_OF_REQUIREMENT,
        }
    answers = await _ask(scorer, api_key, _posting_state(job_text), questions)
    return [answers[f"k{number}"]["choice"] for number in range(len(lines))]


# ══ Step 8: the match score ═════════════════════════════════════════════════════
#
# The score follows the usual shape of an ATS score: a few parts, each scored on its own, added by weight.
# Every part is judged by Jev from the posting's own lines, so the same work in different words still counts.
#
#     skills            how much of the skill lines the résumé covers (must-haves count double)
#     responsibilities  how much of the duties the candidate has actually done
#     experience        whether the candidate has the years and level, counting only this kind of work
#     qualifications    how much of the degree and certification lines the résumé covers
#
# A part the posting says nothing about is left out and its weight is shared among the others.
#
# Last, role fit scales the total: whether the résumé shows the candidate doing this kind of work.
# Without it, a data engineer who lists the same tools scores like a DevOps engineer.
#     score = role fit x (a fixed part + the rest x the total of the parts)

SKILLS = "skills"
RESPONSIBILITIES = "responsibilities"
EXPERIENCE_PART = "experience"
QUALIFICATIONS = "qualifications"
WEIGHT_OF_PART = {SKILLS: 0.45, RESPONSIBILITIES: 0.25, EXPERIENCE_PART: 0.15, QUALIFICATIONS: 0.15}

ROLE_FIT_QUESTION = "core_match"
EXPERIENCE_QUESTION = "experience_match"
# In a part, a must-have line counts twice as much as a preferred one.
MUST_HAVE_COUNTS_FOR = 2
PREFERRED_COUNTS_FOR = 1
# The part of the role fit a résumé keeps when the parts add up to nothing.
PART_KEPT_WITH_NO_COVERAGE = 0.4


async def match_score(scorer: str, resume_text: str, job_text: str) -> dict | None:
    """How well the résumé matches the posting, out of 100, with the parts and the lines behind it.

    Returns {"score", "role_fit", "total_of_parts", "parts": {part: share or None}, "must_have": [rows],
    "preferred": [rows], "duties": [rows], "found_from"}. A must-have or preferred row is
    {"line", "covered", "weight", "kind"}; a duty row is {"line", "covered"}.
    The score is None when the posting gave no lines to score. Returns None when Jev cannot be asked.
    """
    lists = await lines_of_the_posting_by_what_they_are(scorer, job_text)
    if lists is None:
        return None

    api_key = decision_scorer._find_api_key(scorer)
    requirement_lines = lists["must_have"] + lists["preferred"]
    try:
        covered, duties_done, kinds, role_answers = await asyncio.gather(
            _how_much_of_each_line_is_covered(scorer, api_key, resume_text, job_text, requirement_lines),
            _how_much_of_each_duty_is_done(scorer, api_key, resume_text, job_text, lists["duties"]),
            _kind_of_each_line(scorer, api_key, job_text, requirement_lines),
            decision_scorer.score_resume(scorer, resume_text, posting_as_plain_text(job_text)))
    except Exception as error:
        logger.warning(f"The resume could not be scored against the posting ({type(error).__name__}: {error})")
        return None
    if role_answers is None:
        return None

    must_have, preferred = _requirement_rows(lists, covered, kinds)
    duties = [{"line": line, "covered": round(share, 2)} for line, share in zip(lists["duties"], duties_done)]
    has_lines_to_score = bool(must_have or preferred or duties)
    parts = {
        SKILLS: _share_of_kind(must_have, preferred, A_SKILL),
        RESPONSIBILITIES: _plain_average([row["covered"] for row in duties]),
        EXPERIENCE_PART: role_answers["answers"][EXPERIENCE_QUESTION] if has_lines_to_score else None,
        QUALIFICATIONS: _share_of_kind(must_have, preferred, A_QUALIFICATION),
    }
    role_fit = role_answers["answers"][ROLE_FIT_QUESTION]
    total_of_parts = _parts_added_by_weight(parts)
    return {"score": _score_out_of_100(role_fit, total_of_parts), "role_fit": round(role_fit, 2),
            "total_of_parts": total_of_parts,
            "parts": {part: (None if share is None else round(share, 2)) for part, share in parts.items()},
            "must_have": must_have, "preferred": preferred, "duties": duties, "found_from": lists["found_from"]}


def _requirement_rows(lists: dict, covered: list, kinds: list) -> tuple:
    """The must-have rows and the preferred rows, each with its coverage, its weight and its kind."""
    how_many_must_haves = len(lists["must_have"])
    must_have_weights = _weights_so_each_tool_counts_once(lists["must_have"])
    preferred_weights = _weights_so_each_tool_counts_once(
        lists["preferred"], tools_counted_already=_tools_named_in_any_of(lists["must_have"]))
    must_have = _lines_with_their_coverage(lists["must_have"], covered[:how_many_must_haves], must_have_weights)
    preferred = _lines_with_their_coverage(lists["preferred"], covered[how_many_must_haves:], preferred_weights)
    for row, kind in zip(must_have + preferred, kinds):
        row["kind"] = kind
    return must_have, preferred


def _share_of_kind(must_have: list, preferred: list, kind: str):
    """How much of the lines of one kind the résumé covers; None when the posting has none of that kind."""
    must_have_covered = _average_coverage([row for row in must_have if row["kind"] == kind])
    preferred_covered = _average_coverage([row for row in preferred if row["kind"] == kind])
    return _coverage_of_both_lists(must_have_covered, preferred_covered)


def _coverage_of_both_lists(must_have_covered, preferred_covered):
    """One coverage figure from the two averages; a list with nothing in it is left out."""
    counted = []
    if must_have_covered is not None:
        counted.append((must_have_covered, MUST_HAVE_COUNTS_FOR))
    if preferred_covered is not None:
        counted.append((preferred_covered, PREFERRED_COUNTS_FOR))
    if not counted:
        return None
    total = sum(counts_for for _covered, counts_for in counted)
    return sum(covered * counts_for for covered, counts_for in counted) / total


def _plain_average(shares: list):
    return sum(shares) / len(shares) if shares else None


def _parts_added_by_weight(parts: dict):
    """The parts added by their weights; a part with nothing to score gives its weight to the others."""
    scored_parts = {part: share for part, share in parts.items() if share is not None}
    if not scored_parts:
        return None
    total_weight = sum(WEIGHT_OF_PART[part] for part in scored_parts)
    return round(sum(share * WEIGHT_OF_PART[part] for part, share in scored_parts.items()) / total_weight, 2)


def _score_out_of_100(role_fit: float, total_of_parts):
    if total_of_parts is None:
        return None
    part_earned = (1 - PART_KEPT_WITH_NO_COVERAGE) * total_of_parts
    return round(100 * role_fit * (PART_KEPT_WITH_NO_COVERAGE + part_earned))
