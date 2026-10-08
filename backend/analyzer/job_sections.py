"""Splits a job posting into its sections with patterns only: no model is involved, so the same posting always splits the same way."""
import re

from backend.analyzer.job_requirements import clean_job_text
from backend.analyzer.known_skills import KNOWN_SKILLS

MUST_HAVE = "must_have"
NICE_TO_HAVE = "nice_to_have"
RESPONSIBILITIES = "responsibilities"
ABOUT = "about"
GENERAL = "general"
IGNORED = "ignored"

HEADING_MAX_CHARS = 70

# What a heading says, by the section it opens. Checked in this order, so "Preferred
# Qualifications" is read as nice-to-have before "Qualifications" can claim it.
HEADING_PATTERNS = [
    (IGNORED, r"benefit|perks|why join|our culture|our values|^values$|advantages of joining|equal opportunity|"
              r"security alert|how we look after|candidate submission|internal applicants|how to apply|"
              r"compensation|salary range|^pay\b"),
    # A job board's "Preferred candidate profile" is its name for the requirements.
    (MUST_HAVE, r"candidate profile"),
    (NICE_TO_HAVE, r"preferred|nice[- ]to[- ]have|good[- ]to[- ]have|bonus|desired skills|desirable|"
                   r"pluses|\bplus\b|would be great|optional"),
    (MUST_HAVE, r"requirement|required|must[- ]have|qualification|what you have|what you bring|"
                r"what you.ll need|what we.re looking for|who you are|key skills|skills required|"
                r"mandatory|typically have|what it takes|ideal candidate|key technologies|competenc"),
    (RESPONSIBILITIES, r"responsibilit|what you.ll do|what you will do|you will$|what you.ll own|"
                       r"your role|the role$|role description|day[- ]to[- ]day|duties|expectations"),
    (ABOUT, r"^about\b|who we are|our company|organizational overview|company overview|^overview$"),
    (GENERAL, r"job description|job summary|role summary|about the role|position summary|^education"),
]

NICE_TO_HAVE_WORDS = re.compile(
    r"\b(preferred|nice[- ]to[- ]have|good[- ]to[- ]have|bonus|a plus|is a plus|desirable|ideally|optional|"
    r"an advantage|would be (?:great|helpful))\b", re.I)

# A heading written inside a paragraph, such as "... AI platforms. Key Responsibilities Design and deliver ...".
INLINE_HEADING = re.compile(
    r"(?:(?<=[.!?:])\s+|^)"
    r"(Key Responsibilities|Primary Responsibilities|Roles? (?:&|and) Responsibilities|Responsibilities|"
    r"Required Qualifications?|Preferred Qualifications?|Basic Qualifications?|Required Skills|Preferred Skills|"
    r"Must[- ]Have Skills|Good[- ]to[- ]Have Skills|Nice[- ]to[- ]Have Skills|Qualifications|Requirements)"
    r":?\s+(?=[A-Z])", re.M)

LONG_LINE_CHARS = 120
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

# Lines that describe the posting itself and not the work or the candidate.
POSTING_DETAIL_LABEL = re.compile(
    r"^(role|role category|industry type|department|employment type|location|job location|work (?:mode|model|"
    r"timings?|shift|location)|job (?:type|id|family)|category|date posted|reporting to|size of the team|"
    r"on-?site|level|grade|pay|salary|function|experience level|experience|exp|notice period|domain)\b"
    r"[^.]{0,40}[:\-–]",
    re.I)
# A line a job board adds to every posting.
JOB_BOARD_NOTE = re.compile(r"skills highlighted with", re.I)
# A skill list that names its own group: "Primary Skill : DevOps, Azure" or "Secondary: Terraform".
REQUIRED_SKILLS_LABEL = re.compile(
    r"^(primary|mandatory|must[- ]have|required|key|technical|core)?\s*skills?(?: required)?\s*:", re.I)
OPTIONAL_SKILLS_LABEL = re.compile(
    r"^(secondary|good[- ]to[- ]have|nice[- ]to[- ]have|preferred|optional|additional)(?: skills?)?\s*:", re.I)
LEGAL_OR_BENEFIT_WORDS = re.compile(
    r"equal opportunity|reasonable accommodation|without regard to|protected characteristic|"
    r"background check|benefits package|health insurance|salary|compensation|pay transparency|"
    r"in exchange for employment", re.I)
TRACKING_CODE = re.compile(r"^#?[A-Z]{2,4}-[A-Za-z0-9]+$")

# A sentence that introduces a list, such as "you will bring the following skillset & experience:".
LIST_LEAD_IN_MAX_CHARS = 200
LIST_LEAD_IN_PATTERNS = [
    (NICE_TO_HAVE, r"nice to have|good to have|preferred|bonus|a plus|not required|desirable"),
    (MUST_HAVE, r"skillset|skills|experience|qualif|requirement|you will bring|you.ll bring|you bring|"
                r"you have|you.ll need|you will need|looking for"),
    (RESPONSIBILITIES, r"you will|you.ll|responsibilit|duties|day[- ]to[- ]day"),
]
# A job board's own skill tags, printed stuck together, repeat what the posting already says.
LONGEST_READABLE_TAG_LINE_CHARS = 40
# A sentence about the employer that has drifted into a list of requirements or duties.
EMPLOYER_PITCH_OPENING = re.compile(r"^(We |We.re |Our |At [A-Z]|Today|Join us|As a (?:global|leading))")
WORDS_A_TITLE_KEEPS_SMALL = {"and", "or", "of", "to", "the", "in", "for", "a", "an", "&", "/", "-"}
SUB_HEADING_MAX_WORDS = 6
LEADING_LABEL = re.compile(r"^[A-Z][^:.]{2,60}:\s+")

# How a line reads when no heading says what it is.
DUTY_OPENING = re.compile(
    r"^(you will|you.ll|responsible for|acts? as|architect|design|develop|build|lead|drive|own|ensure|collaborate|"
    r"contribute|mentor|support|manage|implement|create|maintain|review|generate|promote|advocate|deliver|"
    r"work with|work closely|partner|define|identify|translate|provide|conduct|analy[sz]e|optimi[sz]e|"
    r"monitor|automate|deploy|troubleshoot|participate|coordinate|establish|evaluate|write|apply|use)\w*\b", re.I)
REQUIREMENT_WORDS = re.compile(
    r"\b(experience (?:with|in|as|of|working|building|developing|designing)|previous experience|knowledge of|"
    r"understanding of|proficien\w+|expertise in|background in|ability to|hands[- ]on|familiarity with|"
    r"degree\b|certifi\w+|\d{1,2}\s*\+?\s*(?:years?|yrs?)|must have|strong [\w-]+ skills|skilled in|"
    r"track record)", re.I)

# "5+ years", "1-5 years", "2 to 3 yrs" or plain "5 years".
YEARS_PATTERN = re.compile(
    r"(?P<low>\d{1,2})\s*(?P<or_more>\+|plus)?\s*(?:(?:-|–|to)\s*(?P<high>\d{1,2}))?\s*(?P<or_more_after>\+)?\s*"
    r"(?:years?|yrs?)\b", re.I)
# "Exp: 6 to 12" or "Experience - 5+", where the label says the number is years.
LABELLED_YEARS_PATTERN = re.compile(
    r"^(?:total |work |relevant )?exp(?:erience)?\b[^:\-–\d]{0,25}[:\-–]\s*(?P<low>\d{1,2})\s*(?P<or_more>\+)?\s*"
    r"(?:(?:-|–|to)\s*(?P<high>\d{1,2}))?\s*(?P<or_more_after>\+)?", re.I)
MAX_SENSIBLE_YEARS = 40

EDUCATION_WORDS = re.compile(
    r"\b([Bb]achelor|[Mm]aster'?s\b|Ph\.?D|[Dd]octorate|B\.?\s?Tech|M\.?\s?Tech|B\.?E\b|B\.?S\b|M\.?S\b|MBA|"
    r"[Dd]egree in|[Gg]raduate\b|[Pp]ost[- ]?[Gg]raduate|[Dd]iploma)")
CERTIFICATION_WORDS = re.compile(r"\b(certified|certification|certificate|licen[sc]e)", re.I)
WORK_AUTHORIZATION_WORDS = re.compile(
    r"\b(authori[sz]ed to work|work authori[sz]ation|visa|sponsorship|citizen|green card|"
    r"security clearance|right to work)", re.I)
EXPERIENCE_LEVEL_WORDS = re.compile(
    r"\b(years?|yrs?|senior|junior|lead|principal|staff|entry[- ]level|mid[- ]level|experience level|seniority)\b",
    re.I)

BULLET_MARKS = "-•*+●◦▪·"


def _heading_kind(line: str) -> str | None:
    """The section a line opens, or None when the line is ordinary text."""
    is_bullet = line[:1] in BULLET_MARKS
    # "Location: As per requirement" is a labelled value, not a heading.
    is_labelled_value = ":" in line.rstrip(":")
    if is_bullet or is_labelled_value or len(line) > HEADING_MAX_CHARS:
        return None
    wording = line.rstrip(":").strip().lower()
    for kind, pattern in HEADING_PATTERNS:
        if re.search(pattern, wording):
            return kind
    return None


def _put_inline_headings_on_their_own_line(posting: str) -> str:
    return INLINE_HEADING.sub(lambda match: f"\n{match.group(1)}\n", posting)


def _sentences(line: str) -> list:
    """A long paragraph as its sentences; anything shorter is one item already."""
    if len(line) <= LONG_LINE_CHARS:
        return [line]
    return [sentence.strip() for sentence in SENTENCE_BREAK.split(line) if sentence.strip()]


def _is_about_the_posting(line: str) -> bool:
    """A detail of the posting ("Role: DevOps Engineer"), a sub-heading or a tracking code: not content."""
    is_label_without_value = line.endswith((":", "-", "–")) and len(line) <= HEADING_MAX_CHARS
    is_shouted_heading = line.isupper() and len(line) <= HEADING_MAX_CHARS
    # "DevOps Senior Engineer | Hyderabad | 10-15 Years" is the posting's title bar.
    is_title_bar = line.count(" | ") >= 2 and len(line) <= 2 * HEADING_MAX_CHARS
    return bool(is_label_without_value or is_shouted_heading or is_title_bar
                or POSTING_DETAIL_LABEL.search(line) or TRACKING_CODE.match(line))


def _is_a_sub_heading(line: str) -> bool:
    """A short title inside a list of duties, such as "Monitoring & Reliability": it names a group, not a duty."""
    words = [word for word in line.split() if word.lower() not in WORDS_A_TITLE_KEEPS_SMALL]
    is_short = 0 < len(words) <= SUB_HEADING_MAX_WORDS and not line.endswith((".", ",", ";"))
    return is_short and all(word[:1].isupper() for word in words)


def _list_lead_in_kind(line: str) -> str | None:
    """The section a lead-in sentence opens ("... you will bring the following skillset & experience:")."""
    if not line.endswith(":") or len(line) > LIST_LEAD_IN_MAX_CHARS:
        return None
    wording = line.lower()
    for kind, pattern in LIST_LEAD_IN_PATTERNS:
        if re.search(pattern, wording):
            return kind
    return None


def _is_a_list_item(raw_line: str) -> bool:
    return raw_line[:1] in " \t" or raw_line.strip()[:1] in BULLET_MARKS


def _lines_by_section(posting: str) -> dict:
    sections = {kind: [] for kind, _ in HEADING_PATTERNS}
    sections["posting_details"] = []
    current = GENERAL
    list_has_started = False
    reading_job_board_tags = False
    for raw_line in _put_inline_headings_on_their_own_line(posting).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if JOB_BOARD_NOTE.search(line):
            reading_job_board_tags = True
            continue
        is_list_item = _is_a_list_item(raw_line)
        kind = _heading_kind(line) or (None if is_list_item else _list_lead_in_kind(line))
        if kind:
            current, list_has_started = kind, False
            continue
        if reading_job_board_tags and len(line) > LONGEST_READABLE_TAG_LINE_CHARS:
            sections["posting_details"].append(line)
            continue
        is_stray_paragraph = (list_has_started and not is_list_item
                              and current in (MUST_HAVE, NICE_TO_HAVE, RESPONSIBILITIES)
                              and _is_a_paragraph_about_other_matters(line))
        list_has_started = list_has_started or is_list_item
        for item in _sentences(line.lstrip(BULLET_MARKS + " ").strip()):
            sections[IGNORED if is_stray_paragraph else _section_for(item, current)].append(item)
    return sections


def _is_a_paragraph_about_other_matters(line: str) -> bool:
    """A long line that is not a list item and reads as neither a duty nor a requirement, such as a culture statement."""
    reads_as_content = DUTY_OPENING.match(line) or REQUIREMENT_WORDS.search(line)
    return len(line) > HEADING_MAX_CHARS and not reads_as_content


def _section_for(item: str, section_of_its_heading: str) -> str:
    """Where one line goes: what the line says about itself comes before the heading it sits under."""
    if LEGAL_OR_BENEFIT_WORDS.search(item):
        return IGNORED
    if OPTIONAL_SKILLS_LABEL.match(item):
        return NICE_TO_HAVE
    if REQUIRED_SKILLS_LABEL.match(item):
        return MUST_HAVE
    if _is_about_the_posting(item):
        return "posting_details"
    return section_of_its_heading


def _file_unheaded_lines_by_how_they_read(sections: dict) -> None:
    """A line under no heading goes where its wording points: a requirement, a duty, or it stays general."""
    still_general = []
    for line in sections[GENERAL]:
        wording = LEADING_LABEL.sub("", line)
        if EMPLOYER_PITCH_OPENING.match(line):
            still_general.append(line)
        elif REQUIREMENT_WORDS.search(wording):
            sections[MUST_HAVE].append(line)
        elif DUTY_OPENING.match(wording):
            sections[RESPONSIBILITIES].append(line)
        else:
            still_general.append(line)
    sections[GENERAL] = still_general


def _move_requirements_out_of_the_duties(sections: dict) -> None:
    """A line under a duties heading that reads as a requirement ("Strong experience in ...") is a requirement."""
    still_duties = []
    for line in sections[RESPONSIBILITIES]:
        wording = LEADING_LABEL.sub("", line)
        reads_as_a_requirement = REQUIREMENT_WORDS.search(wording) and not DUTY_OPENING.match(wording)
        (sections[MUST_HAVE] if reads_as_a_requirement else still_duties).append(line)
    sections[RESPONSIBILITIES] = still_duties


def _take_out_lines_that_are_not_content(sections: dict) -> None:
    """Employer pitch goes to the about section; sub-headings inside the duties are dropped."""
    for kind in (MUST_HAVE, NICE_TO_HAVE, RESPONSIBILITIES):
        pitch = [line for line in sections[kind] if EMPLOYER_PITCH_OPENING.match(line)]
        sections[ABOUT].extend(pitch)
        sections[kind] = [line for line in sections[kind] if line not in pitch]
    sub_headings = [line for line in sections[RESPONSIBILITIES] if _is_a_sub_heading(line)]
    sections["posting_details"].extend(sub_headings)
    sections[RESPONSIBILITIES] = [line for line in sections[RESPONSIBILITIES] if line not in sub_headings]


def _move_optional_lines_out_of_must_haves(sections: dict) -> None:
    """A line that calls itself preferred is nice-to-have, whatever heading it sits under."""
    still_required = []
    for line in sections[MUST_HAVE]:
        # "(Azure preferred)" marks one option inside the line, not the whole line.
        outside_brackets = re.sub(r"\([^)]*\)", "", line)
        target = sections[NICE_TO_HAVE] if NICE_TO_HAVE_WORDS.search(outside_brackets) else still_required
        target.append(line)
    sections[MUST_HAVE] = still_required


def _years_as_stated(match: re.Match) -> str:
    """The figure the way the posting gives it: "1-5", "5+" or "5"."""
    low, high = match.group("low"), match.group("high")
    if high:
        return f"{low}-{high}"
    asks_for_more = match.group("or_more") or match.group("or_more_after")
    return f"{low}+" if asks_for_more else low


def _years_asked_for(lines: list) -> str | None:
    """The first years figure these lines state; a posting leads with its overall requirement."""
    for line in lines:
        matches = [*YEARS_PATTERN.finditer(line), *filter(None, [LABELLED_YEARS_PATTERN.match(line)])]
        for match in matches:
            if 0 < int(match.group("low")) <= MAX_SENSIBLE_YEARS:
                return _years_as_stated(match)
    return None


def _lines_matching(lines: list, pattern: re.Pattern) -> list:
    return [line for line in lines if pattern.search(line)]


def _skill_pattern(name: str) -> re.Pattern:
    # A short name such as "Go" or "R" must match its capitals exactly, or it matches ordinary words.
    flags = re.I if len(name) > 3 else 0
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", flags)


def known_skills_in(lines: list) -> list:
    """The known skills named in these lines, under their standard names, in the list's own order."""
    text = "\n".join(lines)
    return [canonical for canonical, aliases in KNOWN_SKILLS.items()
            if any(_skill_pattern(name).search(text) for name in (canonical, *aliases))]


def split_posting(job_text: str) -> dict:
    """A posting's sections as lists of lines, plus the facts patterns can read from it."""
    sections = _lines_by_section(clean_job_text(job_text))
    _file_unheaded_lines_by_how_they_read(sections)
    _take_out_lines_that_are_not_content(sections)
    _move_requirements_out_of_the_duties(sections)
    _move_optional_lines_out_of_must_haves(sections)

    stated_anywhere = [line for kind in (MUST_HAVE, NICE_TO_HAVE, RESPONSIBILITIES, GENERAL)
                       for line in sections[kind]]
    must_have_skills = known_skills_in(sections[MUST_HAVE])
    nice_to_have_skills = [skill for skill in known_skills_in(sections[NICE_TO_HAVE])
                           if skill not in must_have_skills]
    return {
        **sections,
        "years": (_years_asked_for(sections["posting_details"] + sections[MUST_HAVE])
                  or _years_asked_for(stated_anywhere)),
        "education": _lines_matching(stated_anywhere, EDUCATION_WORDS),
        "certifications": _lines_matching(stated_anywhere, CERTIFICATION_WORDS),
        "work_authorization": _lines_matching(stated_anywhere, WORK_AUTHORIZATION_WORDS),
        "experience_level": _lines_matching(stated_anywhere, EXPERIENCE_LEVEL_WORDS),
        "must_have_skills": must_have_skills,
        "nice_to_have_skills": nice_to_have_skills,
    }
