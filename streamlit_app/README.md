# JobNavigator — Streamlit résumé shelf

A small second frontend for the two things a résumé shelf is for: looking at your
résumés, and tailoring one to a job description. It stores nothing itself — every read
and write goes to the same FastAPI backend the React frontend uses, so both see one
database and a copy tailored here shows up there.

## What it does

- **Add a résumé** takes a PDF (up to 10 MB), and the backend structures it into a base
  résumé — the same import the React frontend uses, so it shows up in both.
- Lists your base résumés by name. Nothing renders until you click one.
- Tailored copies live behind a dropdown next to the original's name, showing how many
  there are. Clicking the name opens the original; opening the dropdown and clicking an
  entry opens that copy.
- **Download PDF** sits above whatever is open, original or tailored, and saves it under
  the same filename the React frontend would use.
- Either can be deleted with the bin beside it. Deleting takes two clicks, and deleting an
  original says how many tailored copies will go with it — the backend cascades.
- **Tailor to a job description** opens a text box. Paste the posting, press Tailor,
  and the new copy appears under the original when the run finishes (10–30s).
- **ATS score** sits above the preview. A tailored copy is scored against the posting it was
  tailored for; an original is scored against a posting you paste. It shows the score, how much
  of the must-have and preferred lines is covered, and each line with its coverage, gaps first.
- Every preview renders in **Word Classic**, whatever template the résumé has stored.
- An **engine toggle** in the sidebar switches both résumé jobs between line-ups:
  OpenRouter (deepseek-v4.1-flash structures, gemma-4-31b tailors), Claude Haiku 5.5 through
  OpenRouter (both jobs), Codex (gpt-5.6-luna for both) and Claude Code (sonnet-5.5 for both). It writes through `PATCH /api/settings`, the same endpoint the
  React frontend uses, so switching here switches there too.
- Whatever it just made shows **how long it took** and which engine made it. For tailoring
  that is the backend's own `duration_seconds` rather than wall-clock, so the polling
  interval is not counted; import is synchronous, so there the wait *is* the time.

Searches, jobs, applications, settings, PDF export and template switching are not here.
Use the React frontend for those.

## Running it

```bash
cd streamlit_app
pip install -r requirements.txt
API_BASE_URL=http://localhost streamlit run app.py
```

`API_BASE_URL` is where the backend lives. Locally with the compose stack that is
`http://localhost` (Caddy). It can also go in `.streamlit/secrets.toml` — copy
`secrets.toml.example` and fill it in.

`API_KEY` is only needed once a dashboard password is set in the main app's Settings.
Both can also be typed into the sidebar at runtime, which is handy for pointing the same
app at a local and a deployed backend.

## Deploying

The app is a normal Streamlit app: point Streamlit Community Cloud at this directory and
put `API_BASE_URL` (and `API_KEY` if set) in the app's Secrets box. The backend needs to
be reachable from the Streamlit server over HTTPS.

Requests are made server-side by Streamlit, not from the viewer's browser, so CORS on the
backend does not come into it. The résumé preview is embedded as sandboxed HTML rather
than an iframe pointing at the backend, so the browser never needs to reach the API
directly and the API key never leaves the server.

## Endpoints it uses

| Call | Why |
|---|---|
| `GET /api/resumes/shelf` | base résumés with their tailored copies already grouped |
| `POST /api/resumes/import-pdf` | add a résumé from a PDF (runs the LLM inline, so it is slow) |
| `GET /api/resumes/{id}/pdf?template=word` | the PDF to download, named by the backend |
| `DELETE /api/resumes/{id}` | remove one; a base cascades to its tailored copies |
| `GET /api/settings` | which engine the backend is on |
| `PATCH /api/settings` | switch engine (validated server-side, same as the React app) |
| `GET /api/resumes/{id}/preview?template=word` | one résumé as HTML, forced to Word Classic |
| `POST /api/resumes/tailor` | start a run from `{base_resume_id, job_description}` |
| `GET /api/monitor/run/{run_id}` | poll until the run finishes |

The `template` parameter on `/preview` was added for this app and mirrors the one `/pdf`
already had. It overrides the render only — the résumé's stored template is untouched, so
opening one here never changes how it looks in the React frontend.
