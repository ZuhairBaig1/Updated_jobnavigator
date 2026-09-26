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

import api

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
# Seeded once from secrets/env. The widgets below own these keys afterwards, so they
# are set here rather than passed as `value=`, which Streamlit warns about on rerun.
st.session_state.setdefault("api_base_url", api.base_url())
st.session_state.setdefault("api_key", api._secret("API_KEY"))


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
            gone = {resume_id}
            st.session_state.confirm_delete = None
            if st.session_state.selected in gone:
                st.session_state.selected = None
            st.session_state.open_copies -= gone
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


# ── sidebar: where the backend is ───────────────────────────────────────────────
with st.sidebar:
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
        st.warning("The backend is on neither preset. Picking one below will set both jobs.",
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
                  help="Only needed once a dashboard password is set in Settings.")
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
                            run_id = api.start_tailoring(base_id, jd.strip())
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

        try:
            html = api.get_preview_html(selected, PREVIEW_TEMPLATE)
        except api.BackendError as e:
            st.error(str(e))
        else:
            _render_document(html)
