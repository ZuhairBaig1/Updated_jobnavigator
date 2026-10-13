"""JobNavigator — résumé shelf and tailoring, as a Streamlit app.

Deliberately small. It does the two things a résumé shelf is for:

  * show the résumés you have, and what one looks like when you click it
  * tailor one to a job description you paste, and keep the result under the original

Everything else the React frontend does — searches, jobs, applications, settings —
is not here. Previews always render in Word Classic, whatever the résumé has stored,
because this app is for reading the document rather than designing it.
"""
import time

import base64

import streamlit as st

import importlib

import api

# After a deploy Streamlit re-runs this script but keeps the modules it had already imported, so a new
# app.py met an old api.py and failed on a name that file did have. api holds no state, so it is loaded afresh.
api = importlib.reload(api)

PREVIEW_TEMPLATE = "word"          # "Word Classic": the only look this app shows
POLL_SECONDS = 2
POLL_LIMIT = 150                   # 5 minutes. Measured: 7-9s on gemma, 69s on Claude Sonnet
                                   # (which sometimes answers in prose and is asked again).

st.set_page_config(page_title="JobNavigator Résumés", page_icon="📄", layout="wide")


# ── state ───────────────────────────────────────────────────────────────────────
# `selected` is the résumé whose preview is on screen, `tailor_for` is the base whose
# JD box is open. Both hold ids rather than objects so a refreshed shelf never leaves
# a stale copy of a résumé behind.
st.session_state.setdefault("selected", None)
st.session_state.setdefault("tailor_for", None)
# Which bases have their tailored copies showing. Ids, so a refreshed shelf cannot
# leave the list open against a résumé that is no longer there.
st.session_state.setdefault("open_copies", set())
# The résumé a delete has been asked for but not yet confirmed. Deleting a base takes its
# tailored copies with it, so nothing is removed on a single click.
st.session_state.setdefault("confirm_delete", None)
# A one-shot note to show after the rerun that follows a delete.
st.session_state.setdefault("flash", None)
# resume id -> {"seconds", "engine", "job"}. Kept per session rather than stored, since it
# describes this run rather than the résumé: the same copy regenerated on another engine
# takes a different time.
st.session_state.setdefault("timings", {})
# resume id -> the ATS score last worked out for it in this session.
st.session_state.setdefault("ats_scores", {})
# Seeded once from secrets/env. The widgets below own these keys afterwards, so they
# are set here rather than passed as `value=`, which Streamlit warns about on rerun.
# api_key is deliberately left blank here rather than seeded from the secret — the sidebar
# field is only a manual override, and pre-filling it would let anyone viewing the app
# reveal the real backend key through the password field's show icon. api._headers()
# falls back to the secret itself when this field is empty.
st.session_state.setdefault("api_base_url", api.base_url())
st.session_state.setdefault("api_key", "")


@st.cache_data(show_spinner=False, max_entries=32)
def _pdf_bytes(resume_id: str, template: str, _stamp: str):
    """The résumé as a PDF, cached per résumé and per version.

    Rendering goes through headless Chromium on the backend and takes a second or two,
    and Streamlit reruns the whole script on every click — without a cache the PDF would
    be rebuilt each time the page so much as redraws. `_stamp` is the résumé's
    last-changed time, so editing one invalidates its entry instead of serving the old file.
    """
    return api.get_pdf(resume_id, template)


def _render_document(html: str, height: int = 1150):
    """Show a rendered résumé without letting its CSS touch the page around it.

    The document is a whole HTML page — its own body rules, fonts and @page sizing —
    so it has to live in an iframe or it restyles the app. `st.components.v1.html`
    takes the markup directly and sandboxes it, which is what this needs; `st.iframe`
    only accepts a URL, and the preview endpoint may sit behind an API key that an
    iframe `src` cannot carry. That call is deprecated, so if a future Streamlit drops
    it the same markup goes in as a data URL instead.
    """
    try:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=True)
        return
    except Exception:
        pass
    encoded = base64.b64encode(html.encode("utf-8")).decode("ascii")
    st.iframe(f"data:text/html;base64,{encoded}", height=height)


def _ids_of_its_tailored_copies(resume_id: str) -> set:
    """The tailored copies that are deleted along with an original; none for a copy."""
    for base in bases:
        if base["id"] == resume_id:
            return {copy["id"] for copy in base.get("copies") or []}
    return set()


def _confirm_delete_strip(resume_id: str, what: str, also_removes: int = 0):
    """Ask before deleting, and say what else goes with it.

    Shown in place of the row's controls once its bin has been pressed. A base carries
    its tailored copies down with it on the backend, so the count is spelled out rather
    than left for the user to discover afterwards.
    """
    extra = (f" This also deletes its {also_removes} tailored "
             f"cop{'y' if also_removes == 1 else 'ies'}.") if also_removes else ""
    st.warning(f"Delete {what}?{extra} This cannot be undone.", icon="⚠️")
    yes, no = st.columns(2)
    if yes.button("Delete", key=f"yes-{resume_id}", type="primary", use_container_width=True):
        try:
            result = api.delete_resume(resume_id)
        except api.BackendError as e:
            st.error(str(e))
        else:
            gone = {resume_id} | _ids_of_its_tailored_copies(resume_id)
            st.session_state.confirm_delete = None
            if st.session_state.selected in gone:
                st.session_state.selected = None
            st.session_state.open_copies -= gone
            for gone_id in gone:
                st.session_state.ats_scores.pop(gone_id, None)
            n = result.get("children_deleted") or 0
            st.session_state.flash = (f"Deleted {what}"
                                      + (f" and {n} tailored cop{'y' if n == 1 else 'ies'}" if n else "")
                                      + ".")
            st.rerun()
    if no.button("Cancel", key=f"no-{resume_id}", use_container_width=True):
        st.session_state.confirm_delete = None
        st.rerun()


def _select(resume_id: str, label: str):
    st.session_state.selected = resume_id
    st.session_state.selected_label = label
    st.session_state.tailor_for = None


def _copy_label(copy: dict) -> str:
    """What to call a tailored copy, given they all inherit the original's name.

    A copy made from a pasted JD has no company or role, so the time it was made is
    the only thing that tells two of them apart.
    """
    company, role = (copy.get("company") or "").strip(), (copy.get("role") or "").strip()
    if company or role:
        return " — ".join(p for p in (company, role) if p)
    stamp = (copy.get("updated_at") or "").replace("T", " ")[:16]
    return f"Tailored · {stamp}" if stamp else "Tailored copy"


def _is_a_tailored_copy(resume_id: str) -> bool:
    return any(resume_id == copy["id"] for base in bases for copy in base.get("copies") or [])


WHERE_A_TOOL_IS_NAMED = {"only_listed": "only listed as a skill", "not_named": "not named"}


def _what_holds_a_line_back(line: dict) -> str:
    """Why a line earned less than Jev's coverage: the tools it names that the résumé does not show in its work."""
    if line.get("credit", line["covered"]) >= line["covered"]:
        return ""
    held_back = [f"{tool} ({WHERE_A_TOOL_IS_NAMED[where]})" for tool, where in (line.get("tools") or {}).items()
                 if where in WHERE_A_TOOL_IS_NAMED]
    return ", ".join(held_back)


def _requirement_rows(lines: list) -> list:
    """The lines as table rows, the lowest credit first, so the gaps are at the top."""
    rows = [{"Requirement": line["line"], "Credit": round(100 * line.get("credit", line["covered"])),
             "Coverage found": f"{100 * line['covered']:.0f}%",
             "Held back by": _what_holds_a_line_back(line),
             "Importance (1-8)": line.get("importance")} for line in lines]
    return sorted(rows, key=lambda row: row["Credit"])


def _show_findability(score: dict):
    """Whether a recruiter's search would find the résumé. Shown beside the score, never part of it."""
    findability, title = score.get("findability") or {}, score.get("title") or {}
    if not findability:
        return
    st.markdown("**Findability** — would a recruiter's search turn this resume up?")
    keywords, title_match = st.columns(2)
    coverage = findability.get("keyword_coverage_pct")
    keywords.metric("Posting keywords named", "—" if coverage is None else f"{coverage}%")
    title_match.metric("Title match", f"{100 * title['match']:.0f}%" if title.get("present") else "No title")
    if findability.get("missing_keywords"):
        st.caption("Keywords in the posting that the resume never names: " + ", ".join(findability["missing_keywords"]))
    if title.get("present"):
        st.caption(f"Title match compares \"{title['best_line']}\" with the posting's job title. 100% is word for word.")
    else:
        st.caption("This resume prints no title under the name, so there is nothing to match against the posting's job title.")


def _show_missing_essentials(score: dict):
    """The plain-words verdict: which things the job cannot be done without are poorly shown, and what they cost."""
    for missing in score.get("missing_essentials") or []:
        not_shown = ", ".join(missing.get("tools_not_shown") or [])
        what = f"**{not_shown}**" if not_shown else f"\"{missing['line']}\""
        st.error(f"Missing essential: {what}. The job cannot be done without it, and the resume shows "
                 f"{100 * missing['credit']:.0f}% of that requirement. This takes {100 * missing['cost']:.0f}% off the score.")


PARTS_OF_THE_RUBRIC = {"occupation": "Same occupation", "title": "Title match", "core_skills": "Core skills",
                       "other_skills": "Other skills", "duties": "Duties", "experience": "Experience",
                       "qualifications": "Qualifications", "evidence": "Evidence in the roles"}


def _what_a_part_earned(row: dict, times_asked: int) -> str:
    """What the part earned, saying so when that is an average and differs from what the option shown stands for."""
    if row["share"] is None:
        return "—"
    earned = f"{100 * row['share']:.0f}%"
    shown = row.get("option_share")
    if times_asked > 1 and shown is not None and abs(row["share"] - shown) >= 0.01:
        return f"{earned} (average of {times_asked} runs)"
    return earned


KEYWORD_WEIGHT = "15%"


def _keyword_row(found: dict) -> dict:
    """The keyword match as a row of the table: it is a part of the score like the others, worked out by code."""
    used, listed, missing = (len(found.get(key) or []) for key in ("used", "listed", "missing"))
    share = found.get("share")
    return {"Part": "Keyword match", "Picked": f"{used} of {used + listed + missing} keyword requirements used in a role",
            "Earned": "—" if share is None else f"{100 * share:.0f}%", "Weight": KEYWORD_WEIGHT,
            "Reason": f"Checked by code, word for word: {used} used in a role, {listed} only listed as a skill "
                      f"(each counts 40%), {missing} not in the resume. A choice of keywords counts once, by its best."}


def _show_time_taken(score: dict):
    """How long the score took to work out, as this app waited for it."""
    seconds = score.get("_seconds")
    if seconds is None:
        return
    how = " — remembered from an earlier run of the same resume and posting" if score.get("remembered") else ""
    st.caption(f"⏱ Calculated in {seconds:.1f} seconds{how}")


def _show_rubric_score(score: dict):
    """The score from the rubric: the total, what scaled it, the model's pick and reason per question, and the keywords."""
    total, occupation, core, keywords = st.columns(4)
    total.metric("ATS score", f"{score['ats_score']} / 100")
    occupation.metric("Same occupation", f"{100 * score['same_occupation']:.0f}%")
    core.metric("Core skills kept", f"{100 * score['core_skills_kept']:.0f}%")
    keyword_share = (score.get("keywords") or {}).get("share")
    keywords.metric("Keywords matched", "—" if keyword_share is None else f"{100 * keyword_share:.0f}%")
    st.caption("Score = the parts below added by weight (the weights add up to 100%) × same occupation × core skills kept. "
               "The model picks one option per question and says why; the code does the arithmetic. "
               f"Asked {score.get('times_asked', 1)} times: each part earns the average, and the middle pick is shown.")
    for missing in score.get("missing") or []:
        st.error(f"Missing: {missing}")
    rows = [{"Part": PARTS_OF_THE_RUBRIC.get(row["question"], row["question"]), "Picked": row["meaning"],
             "Earned": _what_a_part_earned(row, score.get("times_asked", 1)),
             "Weight": "scales the total" if row["weight"] is None else f"{100 * row['weight']:.0f}%",
             "Reason": row["reason"]} for row in score.get("rubric") or []]
    rows.append(_keyword_row(score.get("keywords") or {}))
    st.dataframe(rows, hide_index=True, use_container_width=True)
    core = score.get("core_skills") or {}
    if core.get("named"):
        st.markdown("**Core skills** — what the job is mainly about, looked for in the roles")
        st.caption(f"Shown in a role: {', '.join(core.get('used') or []) or 'none'}")
        st.caption(f"Not shown in any role: {', '.join(core.get('not_used') or []) or 'none'}")
    found = score.get("keywords") or {}
    st.markdown("**Keyword match** — the posting's keywords, looked for word for word")
    st.caption(f"Used in a role: {', '.join(found.get('used') or []) or 'none'}")
    st.caption(f"Only listed as a skill: {', '.join(found.get('listed') or []) or 'none'}")
    st.caption(f"Not in the resume: {', '.join(found.get('missing') or []) or 'none'}")
    if found.get("shown_another_way"):
        st.caption("Counted as used because a role shows it under another name: "
                   + ", ".join(f"{item['keyword']} (by {item['shown_by']})" for item in found["shown_another_way"]))
    if found.get("alternatives_not_needed"):
        st.caption("Alternatives the posting offered that were not needed, because another one is met: "
                   + ", ".join(found["alternatives_not_needed"]))
    if score.get("input_tokens"):
        thinking = score.get("thinking_tokens") or 0
        of_which = f" (of which {thinking:,} thinking)" if thinking else ""
        st.caption(f"Tokens for this score, over {score.get('times_asked', 1)} call(s) to {score.get('model')}: "
                   f"{score['input_tokens']:,} in, {score['output_tokens']:,} out{of_which}. "
                   "These are what the model was sent and wrote in full, with anything the route adds and all thinking.")


def _show_ats_score(score: dict):
    _show_time_taken(score)
    if score.get("scorer") == "rubric":
        _show_rubric_score(score)
        return
    if score.get("ats_score") is None:
        st.warning("No must-have or preferred lines were found in this job description, so there is nothing to score.")
        return
    _show_missing_essentials(score)
    total, must_have, preferred, occupation, essentials = st.columns(5)
    total.metric("ATS score (fit)", f"{score['ats_score']} / 100")
    essentials.metric("Essentials kept", f"{100 * score.get('essentials_kept', 1.0):.0f}%")
    must_have_coverage, preferred_coverage = score.get("must_have_coverage"), score.get("preferred_coverage")
    must_have.metric("Must-haves earned", "—" if must_have_coverage is None else f"{100 * must_have_coverage:.0f}%")
    preferred.metric("Preferred earned", "—" if preferred_coverage is None else f"{100 * preferred_coverage:.0f}%")
    occupation.metric("Same occupation", f"{100 * score['same_occupation']:.0f}%")
    st.caption("Score = (80% must-haves + 20% preferred) × same occupation × essentials kept. More important lines count for more. "
               "A tool the resume only lists as a skill, or never names, earns little however much related work is found.")
    _show_findability(score)
    for heading, lines in (("Must-have lines", score.get("must_have")), ("Preferred lines", score.get("preferred"))):
        if lines:
            st.markdown(f"**{heading}**")
            st.dataframe(_requirement_rows(lines), hide_index=True, use_container_width=True,
                         column_config={"Credit": st.column_config.ProgressColumn(
                             "Credit", format="%d%%", min_value=0, max_value=100)})


def _ats_score_panel(resume_id: str):
    """Work out and show the ATS score of the résumé on screen."""
    is_a_copy = _is_a_tailored_copy(resume_id)
    with st.expander("ATS score", expanded=resume_id in st.session_state.ats_scores):
        if is_a_copy:
            st.caption("Scored against the job description this copy was tailored for.")
            job_description = ""
        else:
            job_description = st.text_area("Job description to score against", height=160,
                                           key=f"ats-jd-{resume_id}", placeholder="Paste the full posting here…")
        method = st.radio("Scoring method", list(api.SCORING_METHODS), key=f"ats-method-{resume_id}", horizontal=True,
                          help="Rubric: one model call and a keyword check, quick and cheap. "
                               "Line by line: Jev judges every line of the posting, steadier and slower.")
        if st.button("Calculate ATS score", key=f"ats-go-{resume_id}", type="primary"):
            if not is_a_copy and not job_description.strip():
                st.warning("Paste a job description first.")
            else:
                with st.spinner("Scoring…"):
                    try:
                        started = time.monotonic()
                        score = api.get_ats_score(resume_id, job_description.strip(), method)
                        st.session_state.ats_scores[resume_id] = {**score, "_seconds": time.monotonic() - started}
                    except api.BackendError as e:
                        st.error(str(e))
        if resume_id in st.session_state.ats_scores:
            _show_ats_score(st.session_state.ats_scores[resume_id])


def _when(seconds_since_1970) -> str:
    """A reset time as this computer's clock would show it."""
    if not seconds_since_1970:
        return "an unknown time"
    return time.strftime("%a %d %b, %H:%M", time.localtime(seconds_since_1970))


def _show_a_limit(name: str, window) -> None:
    if not window:
        st.caption(f"{name}: not reported")
        return
    used = max(0.0, min(1.0, window["used"]))
    st.progress(used, text=f"{name}: {100 * used:.0f}% used · resets {_when(window.get('resets_at'))}")


def _claude_plan_usage_panel():
    """The Claude plan's limits and what the latest Claude Code calls used, read when asked for."""
    with st.expander("Claude plan usage"):
        st.caption("For engines that run on the Claude Code login. Checking makes one very small call.")
        if st.button("Check usage", key="claude-usage-check"):
            with st.spinner("Reading the limits…"):
                try:
                    st.session_state.claude_usage = api.get_claude_code_usage()
                except api.BackendError as e:
                    st.error(str(e))
        usage = st.session_state.get("claude_usage")
        if not usage:
            return
        limits = usage.get("limits")
        if limits:
            _show_a_limit("5-hour window", limits.get("five_hour"))
            _show_a_limit("7-day window", limits.get("seven_day"))
            if limits.get("status") and limits["status"] != "allowed":
                st.warning(f"The plan is reporting: {limits['status']}. Calls may be refused until a window resets.")
        else:
            st.caption("The limits could not be read. The backend may have no Claude Code login.")
        calls = usage.get("calls") or []
        if calls:
            st.markdown("**Latest calls on the plan**")
            st.dataframe([{"Job": call["job"], "Model": call["model"], "Tokens in": call["input_tokens"],
                           "Tokens out": call["output_tokens"], "Seconds": call["seconds"]} for call in calls],
                         hide_index=True, use_container_width=True)
        added = usage.get("added_per_call") or {}
        st.caption(f"Tokens in and out are the full counts for each call. Tokens in include about "
                   f"{added.get('always', 0)} that Claude Code adds to every call, and about "
                   f"{added.get('for_a_reply_format', 0)} more when a reply format is asked for, which "
                   "import, tailoring and scoring all do. Tokens out include the model's thinking.")


# ── sidebar: where the backend is ───────────────────────────────────────────────
with st.sidebar:
    _claude_plan_usage_panel()
    st.subheader("Engine")
    try:
        _settings = api.get_settings()
        _current = api.current_engine(_settings)
    except api.BackendError as e:
        st.caption(f"Could not read settings: {e}")
        _settings, _current = {}, ""
    _names = list(api.ENGINES)
    _choice = st.radio("Models", _names,
                       index=_names.index(_current) if _current in _names else 0,
                       format_func=lambda n: api.ENGINES[n]["label"],
                       label_visibility="collapsed",
                       help="Switches both résumé import and tailoring. Saved in the backend, "
                            "so the React frontend follows too.")
    st.caption(api.ENGINES[_choice]["help"])
    if not _current:
        st.warning("The backend is on none of these presets. Picking one below will set both jobs.",
                   icon="⚠️")
    if _choice != _current:
        if st.button(f"Switch to {_choice}", type="primary", use_container_width=True):
            try:
                api.set_engine(_choice)
            except api.BackendError as e:
                st.error(str(e))
            else:
                st.session_state.flash = f"Now using {api.ENGINES[_choice]['label']}."
                st.rerun()
    else:
        st.success(f"Active: {api.ENGINES[_choice]['label']}", icon="✅")

    st.divider()
    st.subheader("Backend")
    st.text_input("API base URL", key="api_base_url",
                  help="Where the FastAPI backend is running.")
    st.text_input("API key", key="api_key", type="password",
                  help="Leave blank to use the key from secrets. Only type one here to "
                       "override it for this session.")
    if st.button("Refresh", use_container_width=True):
        st.rerun()

# ── load the shelf ──────────────────────────────────────────────────────────────
try:
    shelf = api.get_shelf()
except api.BackendError as e:
    st.error(str(e))
    st.caption("Set API_BASE_URL in .streamlit/secrets.toml, or type it in the sidebar.")
    st.stop()

bases = shelf.get("bases") or []

# id -> last-changed time, used as part of the PDF cache key so an edited résumé is
# re-rendered rather than served from a stale cache.
stamps = {}
for _b in bases:
    stamps[_b["id"]] = _b.get("updated_at") or ""
    for _c in _b.get("copies") or []:
        stamps[_c["id"]] = _c.get("updated_at") or ""

left, right = st.columns([1, 2], gap="large")


def _add_resume_box():
    """Upload a PDF and let the backend structure it into a base résumé.

    The same endpoint the React frontend imports through, so a résumé added here is a
    normal base résumé everywhere. The upload key is bumped after a successful import so
    the picker clears itself instead of re-submitting the same file on the next rerun.
    """
    st.session_state.setdefault("upload_round", 0)
    with st.expander("Add a résumé", expanded=not bases):
        pdf = st.file_uploader("Résumé PDF", type=["pdf"], key=f"pdf-{st.session_state.upload_round}",
                               help="Up to 10 MB. The backend reads it and structures it into a base résumé.")
        if pdf is not None and st.button("Import", type="primary", use_container_width=True,
                                         key=f"import-{st.session_state.upload_round}"):
            with st.spinner(f"Reading {pdf.name}… this can take a minute."):
                started = time.monotonic()
                try:
                    created = api.import_pdf(pdf.name, pdf.getvalue())
                except api.BackendError as e:
                    st.error(str(e))
                else:
                    # Import is synchronous — there is no run record to ask, so the time
                    # spent waiting on the request is the time it took.
                    elapsed = time.monotonic() - started
                    st.session_state.upload_round += 1
                    if created.get("id"):
                        st.session_state.timings[created["id"]] = {
                            "seconds": elapsed, "engine": _choice, "job": "Imported"}
                        _select(created["id"], created.get("name") or pdf.name)
                    st.rerun()

# ── left: the shelf ─────────────────────────────────────────────────────────────
with left:
    st.subheader("Résumés")
    if st.session_state.flash:
        st.success(st.session_state.pop("flash"))
    _add_resume_box()
    if not bases:
        st.info("No résumés yet. Add one above.")

    for base in bases:
        base_id, base_name = base["id"], base.get("name") or "Untitled"
        copies = base.get("copies") or []
        is_open = base_id in st.session_state.open_copies

        with st.container(border=True):
            # the original's name, with the dropdown for its tailored copies beside it
            name_col, drop_col, del_col = st.columns([5, 1, 1], vertical_alignment="center")
            if name_col.button(f"**{base_name}**", key=f"base-{base_id}",
                               use_container_width=True,
                               type="primary" if st.session_state.selected == base_id else "secondary"):
                _select(base_id, base_name)
                st.rerun()
            if copies:
                if drop_col.button(f"{'▾' if is_open else '▸'} {len(copies)}",
                                   key=f"drop-{base_id}", use_container_width=True,
                                   help="Tailored copies of this résumé"):
                    st.session_state.open_copies ^= {base_id}
                    st.rerun()
            else:
                drop_col.caption("—")
            if del_col.button("🗑", key=f"del-{base_id}", use_container_width=True,
                              help="Delete this résumé and its tailored copies"):
                st.session_state.confirm_delete = base_id
                st.rerun()

            if st.session_state.confirm_delete == base_id:
                _confirm_delete_strip(base_id, f"“{base_name}”", len(copies))

            if is_open:
                for copy in copies:
                    label = _copy_label(copy)
                    chosen = st.session_state.selected == copy["id"]
                    copy_col, cdel_col = st.columns([6, 1], vertical_alignment="center")
                    if copy_col.button(("▸ " if chosen else "• ") + label,
                                       key=f"copy-{copy['id']}", use_container_width=True,
                                       type="tertiary"):
                        _select(copy["id"], f"{base_name} — {label}")
                        st.rerun()
                    if cdel_col.button("🗑", key=f"del-{copy['id']}", use_container_width=True,
                                       type="tertiary", help="Delete this tailored copy"):
                        st.session_state.confirm_delete = copy["id"]
                        st.rerun()
                    if st.session_state.confirm_delete == copy["id"]:
                        _confirm_delete_strip(copy["id"], f"the tailored copy “{label}”")

            if st.button("Tailor to a job description", key=f"tailor-{base_id}",
                         use_container_width=True):
                st.session_state.tailor_for = base_id
                st.rerun()

            # the JD box, opened under the base it will tailor
            if st.session_state.tailor_for == base_id:
                with st.form(key=f"jd-{base_id}", border=False):
                    jd = st.text_area("Paste the job description", height=220,
                                      key=f"jdtext-{base_id}",
                                      placeholder="Paste the full posting here…")
                    way = st.radio("How to tailor", list(api.TAILORING_WAYS), key=f"way-{base_id}",
                                   captions=[w["help"] for w in api.TAILORING_WAYS.values()])
                    go, cancel = st.columns(2)
                    submitted = go.form_submit_button("Tailor", type="primary",
                                                      use_container_width=True)
                    dropped = cancel.form_submit_button("Cancel", use_container_width=True)

                if dropped:
                    st.session_state.tailor_for = None
                    st.rerun()

                if submitted:
                    if not jd.strip():
                        st.warning("Paste a job description first.")
                    else:
                        try:
                            run_id = api.start_tailoring(base_id, jd.strip(), way)
                        except api.BackendError as e:
                            st.error(str(e))
                            run_id = None

                        if run_id:
                            status = "running"
                            with st.status("Tailoring…", expanded=False) as box:
                                for _ in range(POLL_LIMIT):
                                    time.sleep(POLL_SECONDS)
                                    try:
                                        run = api.get_run(run_id)
                                    except api.BackendError:
                                        continue        # a blip mid-run is not a failure
                                    status = run.get("status") or "running"
                                    if status not in ("running", "queued", "pending"):
                                        break
                                box.update(label=f"Tailoring {status}",
                                           state="complete" if status == "completed" else "error")
                            if status == "completed":
                                # The backend times the run itself; wall-clock here would
                                # also count the polling interval and the rerun.
                                secs = (run or {}).get("duration_seconds")
                                before = {c["id"] for c in copies}
                                try:
                                    fresh = api.get_shelf()
                                    now = next((b for b in fresh.get("bases") or []
                                                if b["id"] == base_id), None)
                                    made = ({c["id"] for c in (now or {}).get("copies") or []}
                                            - before)
                                except api.BackendError:
                                    made = set()
                                for new_id in made:
                                    st.session_state.timings[new_id] = {
                                        "seconds": secs, "engine": _choice, "job": "Tailored"}
                                st.session_state.tailor_for = None
                                st.session_state.open_copies |= {base_id}
                                st.rerun()
                            else:
                                st.error(f"The run ended as “{status}”. "
                                         "Check the backend logs for the reason.")

# ── right: the document ─────────────────────────────────────────────────────────
with right:
    selected = st.session_state.selected
    if not selected:
        st.subheader("Preview")
        st.info("Pick a résumé on the left to see it. "
                "Tailored copies sit under the original they came from.")
    else:
        head, dl = st.columns([3, 1], vertical_alignment="bottom")
        head.subheader(st.session_state.get("selected_label") or "Preview")
        _t = st.session_state.timings.get(selected)
        if _t and _t.get("seconds") is not None:
            head.caption(f"Shown in Word Classic. · {_t['job']} in "
                         f"**{_t['seconds']:.0f}s** on {api.ENGINES[_t['engine']]['label']}"
                         if _t["engine"] in api.ENGINES else
                         f"Shown in Word Classic. · {_t['job']} in {_t['seconds']:.0f}s")
        else:
            head.caption("Shown in Word Classic.")

        # The download is prepared up front so the button saves the file on the first
        # press; Streamlit has no way to fetch bytes in response to a download click.
        try:
            pdf, filename = _pdf_bytes(selected, PREVIEW_TEMPLATE, stamps.get(selected, ""))
        except api.BackendError as e:
            dl.caption("PDF unavailable")
            st.warning(f"Could not build the PDF: {e}")
        else:
            dl.download_button("⬇ Download PDF", data=pdf, file_name=filename,
                               mime="application/pdf", use_container_width=True,
                               type="primary", key=f"dl-{selected}")

        _ats_score_panel(selected)

        try:
            html = api.get_preview_html(selected, PREVIEW_TEMPLATE)
        except api.BackendError as e:
            st.error(str(e))
        else:
            _render_document(html)
