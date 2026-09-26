"""The JSON Schemas a model must answer with: parsing a résumé into json_data, and tailoring one to a job.

Passed to the provider as a real response format (OpenAI `json_schema`, Ollama
`format`, Claude forced tool use) rather than pasted into the prompt, so the
shape is enforced by the decoder instead of requested politely.

Every key here is one the renderer or an editor round-trips: the templates
iterate `skills.items()` and print `header.title` / `experience.date` verbatim,
ResumeSections edits `education.years`, and `header.contact` feeds the persona's
autofill keys instead of re-sniffing the contact strings.
"""
import difflib
import json
import logging
import re

logger = logging.getLogger("jobnavigator.resume_schema")

_STR = {"type": "string"}
_BOOL = {"type": "boolean"}
_STR_LIST = {"type": "array", "items": _STR}


def _obj(props: dict) -> dict:
    """An object node with every property required and no extras — what strict mode demands.

    Strict mode has no notion of an optional key, so "missing" is carried by the
    empty string, which is the convention json_data already uses everywhere.
    """
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def _arr(props: dict) -> dict:
    return {"type": "array", "items": _obj(props)}


# A dated entry: the printed range stays verbatim for the templates, the ISO
# pair is what code sorts and compares on. No `pattern` — strict mode rejects it
# on several providers, so YYYY-MM is stated in the prompt and checked after.
_DATED = {"start": _STR, "end": _STR, "current": _BOOL}

RESUME_SCHEMA_NAME = "resume"

RESUME_JSON_SCHEMA = _obj({
    "header": _obj({
        "name": _STR,
        "title": _STR,
        # The contact line exactly as the résumé prints it — what the templates render.
        "contact_items": _arr({"text": _STR, "url": _STR}),
        # The same facts split out, matching persona's autofill keys 1:1.
        "contact": _obj({
            "first_name": _STR, "last_name": _STR, "email": _STR, "phone": _STR,
            "city": _STR, "state": _STR, "country": _STR,
            "linkedin": _STR, "github": _STR, "portfolio": _STR,
        }),
    }),
    "summary": _STR,
    "experience": _arr({
        "company": _STR, "title": _STR, "location": _STR,
        "date": _STR, **_DATED,
        "description": _STR, "bullets": _STR_LIST,
    }),
    # json_data holds skills as an open {label: "a, b"} map, which strict mode
    # cannot express (arbitrary keys need patternProperties / additionalProperties).
    # The model answers with ordered pairs and normalize_resume folds them back.
    "skills": _arr({"label": _STR, "value": _STR}),
    "education": _arr({
        "school": _STR, "location": _STR, "degree": _STR,
        "years": _STR, **_DATED,
    }),
    "projects": _arr({"name": _STR, "url": _STR, "description": _STR, "bullets": _STR_LIST}),
    "publications": _arr({"title": _STR, "description": _STR, "bullets": _STR_LIST}),
})


def _has_content(value) -> bool:
    """True when a field holds something a reader would see.

    Booleans do not count: `current: false` (or true) on an otherwise blank row is
    schema bookkeeping, not content, and letting it count would keep the row alive.
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_has_content(v) for v in value)
    if isinstance(value, dict):
        return any(_has_content(v) for v in value.values())
    return value is not None


def drop_empty_entries(data: dict) -> dict:
    """A copy of `data` with blank list entries and blank bullets removed.

    Strict-mode schemas have no way to say "this section is absent": every property
    is required, so a model with nothing to report may answer with one fully blank
    entry instead of an empty list. Templates guard sections with `{% if section %}`,
    and a one-element list is truthy however empty that element is, so the heading
    renders over nothing. Pruning here keeps that out of every template at once,
    and takes hand-added blank rows with it.
    """
    out = {}
    for key, value in (data or {}).items():
        if isinstance(value, list):
            kept = []
            for item in value:
                if not _has_content(item):
                    continue
                kept.append(drop_empty_entries(item) if isinstance(item, dict) else item)
            out[key] = kept
        else:
            out[key] = value
    return out


def normalize_resume(data: dict) -> dict:
    """A schema-shaped model reply → the json_data shape the templates and editors expect.

    Only `skills` differs between the two: pairs on the wire, `{label: value}` in
    storage, in the order the résumé printed them. A reply that already sent an
    object (a provider that ignored the schema) is passed through untouched.
    """
    out = dict(data or {})
    skills = out.get("skills")
    if isinstance(skills, list):
        folded = {}
        for pair in skills:
            if not isinstance(pair, dict):
                continue
            label = str(pair.get("label") or "").strip()
            if label:
                folded[label] = str(pair.get("value") or "").strip()
        out["skills"] = folded
    # Keep blank entries out of storage too, so the résumé editor opens on the
    # sections the document actually had rather than a row of empty fields.
    return drop_empty_entries(out)


# ── ATS résumés: structured bases and tailored copies ───────────────────────
# Both hold the ATS sections only, in this order: Header, Professional Summary,
# Technical Skills, Work Experience, Projects, Education, Certifications. The reply
# follows that order (strict mode emits keys in schema order); build_base_resume and
# build_tailored_resume fold it back into json_data.

def _ats_schema(role: dict) -> dict:
    return _obj({
        "header": _obj({
            "name": _STR, "title": _STR, "location": _STR, "email": _STR, "phone": _STR,
            "work_authorization": _STR, "github": _STR, "linkedin": _STR,
        }),
        "professional_summary": _STR,
        # Pairs on the wire for the same reason as `skills` above.
        "technical_skills": _arr({"category": _STR, "skills": _STR}),
        "work_experience": _arr(role),
        "projects": _arr({"name": _STR, "url": _STR, "dates": _STR, "technologies": _STR,
                          "bullets": _STR_LIST}),
        "education": _arr({"degree": _STR, "institution": _STR, "graduation_date": _STR,
                           "grade": _STR}),
        "certifications": _STR_LIST,
    })


TAILOR_SCHEMA_NAME = "tailored_resume"

TAILOR_JSON_SCHEMA = _ats_schema({
    "job_title": _STR, "company": _STR, "location": _STR, "dates": _STR,
    "bullets": _STR_LIST,
})

BASE_SCHEMA_NAME = "base_resume"

# The structurer reads each role's dates itself, so a role also carries the ISO pair
# the printed "Mon YYYY – Mon YYYY" is formatted from (and tailoring reads later).
BASE_RESUME_JSON_SCHEMA = _ats_schema({
    "job_title": _STR, "company": _STR, "location": _STR, "dates": _STR, **_DATED,
    "bullets": _STR_LIST,
})

# Marks a copy the renderer prints in that section order, under those section
# names. "_"-prefixed, so it never reaches a template or an editor as content.
LAYOUT_KEY = "_layout"
ATS_LAYOUT = "ats"

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_ISO_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{1,2}))?$")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")
_HEADER_ORDER = ("location", "email", "phone", "work_authorization", "github", "linkedin")


# A model can answer with characters no résumé prints: one route sent backspaces where
# the source had an em dash, 218 of them in a single bullet.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── Technology names ────────────────────────────────────────────────────────
# A résumé gets a tool's name wrong in two ways: it splits a compound ("Fast API",
# "SQL Alchemy") or it cases it wrongly ("Junit"). A strict ATS keyword matcher scores
# neither against a posting asking for "FastAPI", and the term is usually the one the
# whole role is built around. The canonical spelling below is what gets printed.
#
# Only names whose correct form is closed-up belong here. "SQL Server", "Power BI",
# "Azure Data Factory" and "Google Cloud" are correct WITH the space, so joining them
# would be the same bug in reverse.
_TECH_CANONICAL = (
    # languages and markup
    "JavaScript", "TypeScript", "CoffeeScript", "PowerShell", "MATLAB", "VBScript",
    # web and API frameworks
    "FastAPI", "NestJS", "SvelteKit", "AngularJS", "ASP.NET",
    "GraphQL", "WebSocket", "WebSockets", "OpenAPI", "Swagger",
    # ORMs, validation, async
    "SQLAlchemy", "TypeORM", "Pydantic", "Alembic", "Hibernate", "Mongoose", "Celery",
    # databases and stores
    "PostgreSQL", "MySQL", "MariaDB", "SQLite", "MongoDB", "DynamoDB", "CockroachDB",
    "ClickHouse", "InfluxDB", "Elasticsearch", "OpenSearch", "Firestore", "Firebase",
    "BigQuery", "Redshift", "Snowflake", "Databricks", "DataStax", "CassIO", "Neo4j",
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "pgvector", "RabbitMQ", "ZooKeeper",
    # cloud and infrastructure
    "CloudFront", "CloudWatch", "CloudFormation", "SageMaker", "DocumentDB", "OpenShift",
    "Kubernetes", "Terraform", "Ansible", "ArgoCD", "CircleCI", "GitLab", "GitHub",
    "Bitbucket", "SonarQube", "PagerDuty", "Datadog", "Grafana", "Prometheus", "Nginx",
    # testing
    "JUnit", "TestNG", "Mockito", "JaCoCo", "Cypress", "Selenium", "Playwright", "Postman",
    # data and ML
    "NumPy", "SciPy", "XGBoost", "LightGBM", "CatBoost", "TensorFlow", "PyTorch",
    "Matplotlib", "JupyterLab", "MLflow", "Kubeflow", "OpenCV", "NLTK", "AutoML",
    # LLM and AI tooling
    "LangChain", "LangGraph", "LlamaIndex", "OpenAI", "OpenRouter",
    "Composio", "Docling", "DeepLearning.AI",
    # tools and platforms
    "IntelliJ", "PyCharm", "WebStorm", "DataGrip", "VMware", "VirtualBox", "PowerPoint",
    "SharePoint", "ServiceNow", "Salesforce", "QuickBooks", "WordPress", "Shopify",
    "TeamCity", "Bamboo", "Splunk", "Alteryx", "Informatica", "PowerBuilder",
    "AutoCAD", "SolidWorks", "Wireshark", "Nslookup", "Traceroute", "OAuth", "OpenID",
)

# Names a pattern derived from the canonical spelling cannot reach: the correct form is
# a single lowercase word, or it carries punctuation the writer drops.
_TECH_VARIANTS = {
    "pytest": r"py[\s\-]*test",
    "unittest": r"unit[\s\-]*test(?!ing)",
    "pandas": r"pandas",
    "spaCy": r"spacy",
    "scikit-learn": r"scikit[\s\-]*learn|sci[\s\-]*kit[\s\-]*learn",
    "Node.js": r"node[\s.\-]*js",
    "Next.js": r"next[\s.\-]*js",
    "Nuxt.js": r"nuxt[\s.\-]*js",
    "Vue.js": r"vue[\s.\-]*js",
    "D3.js": r"d3[\s.\-]*js",
    "jQuery": r"j[\s\-]*query",
    "Socket.IO": r"socket[\s.\-]*io",
    ".NET": r"dot[\s\-]*net",
    "macOS": r"mac[\s\-]*os",
    "npm": r"npm",
    "dbt": r"dbt",
    "vLLM": r"vllm",
    "YAML": r"yaml",
    "JSON": r"json",
    "JWT": r"jwt",
    "gRPC": r"grpc",
    "CI/CD": r"ci[\s]*/[\s]*cd",
    # Closed-up compounds a case-boundary split cannot see, because they carry no
    # internal capital: "Firestore" reads as one token, so "Fire store" would survive.
    "Firestore": r"fire[\s\-]*store",
    "Firebase": r"fire[\s\-]*base",
    "Elasticsearch": r"elastic[\s\-]*search",
    "Databricks": r"data[\s\-]*bricks",
    "Datadog": r"data[\s\-]*dog",
    "Snowflake": r"snow[\s\-]*flake",
    "Salesforce": r"sales[\s\-]*force",
    "Wireshark": r"wire[\s\-]*shark",
    "Grafana": r"grafana",
    "Kubernetes": r"kubernetes|k8s",
    # Short forms a case-boundary split cannot derive. Order matters inside each
    # alternation: "postgres[\s-]*sql" has to be tried before bare "postgres", or
    # "Postgres SQL" matches the first word and leaves a stray "sql" behind.
    "PostgreSQL": r"postgres[\s\-]*sql|postgre[\s\-]*sql|postgresql|postgres",
    # Not when the résumé already prints the acronym next to it: "Google Cloud Platform
    # (GCP)" must not become "GCP (GCP)", which then fails to match the reply's "GCP (...)"
    # and gets restored beside it as a duplicate.
    "AWS": r"amazon[\s\-]*web[\s\-]*services(?!\s*\(\s*AWS\s*\))",
    "GCP": r"google[\s\-]*cloud[\s\-]*platform(?!\s*\(\s*GCP\s*\))",
}

# The only names where a lowercase split is more likely to be English than a product:
# "a fast API response" is prose, "Fast API" is the framework. Everything else converts
# whatever its case, so "scikit learn" and "node js" are still repaired.
_TECH_COMMON_FIRST_WORD = {"fast", "open", "light", "deep", "big", "simple", "quick",
                           "smart", "cloud", "web", "spring", "swift", "express", "go"}

_TOKEN_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")

# A whole link or address, left exactly as written. "CI/CD" and "and/or" are not links,
# so a bare slash does not qualify.
_URLISH_RE = re.compile(
    r"\S*(?:https?://|www\.|@|\.com|\.org|\.net|\.io|\.ai|\.dev|\.co\b|\.in\b)\S*",
    re.IGNORECASE)


def _tech_rules():
    """(compiled pattern, canonical spelling, guard) for every name worth repairing."""
    rules = []
    for name in _TECH_CANONICAL:
        parts = _TOKEN_RE.findall(name)
        if len(parts) < 2:
            continue                       # one token: nothing a split could have broken
        sep = r"[\s\-]*" if "." not in name else r"[\s.\-]*"
        rules.append((re.compile(r"\b" + sep.join(re.escape(p) for p in parts) + r"\b",
                                 re.IGNORECASE), name,
                      parts[0].lower() in _TECH_COMMON_FIRST_WORD))
    for name, pattern in _TECH_VARIANTS.items():
        first = (_TOKEN_RE.findall(name) or [""])[0].lower()
        rules.append((re.compile(r"\b(?:" + pattern + r")\b", re.IGNORECASE), name,
                      first in _TECH_COMMON_FIRST_WORD))
    return tuple(rules)


_TECH_RULES = _tech_rules()


def canonicalize_tech(text: str) -> str:
    """A prose string with known technology names spelled the way a posting spells them.

    Never used on a URL or an email: "github.com/fastapi/fastapi" would come back recased
    and the link would no longer be the one the résumé printed. A match that had to cross a
    space or hyphen is only joined when it was capitalised as a name, so "a fast API
    response" stays English while "Fast API" becomes FastAPI.
    """
    if not text:
        return text

    # A link inside a bullet is an address, not prose: "github.com/foo/fastapi-demo" would
    # come back as "GitHub.com/foo/FastAPI-demo" and no longer resolve. The stored `url`
    # fields never reach this function, but a bullet may print one.
    if _URLISH_RE.search(text):
        out, last = [], 0
        for m in _URLISH_RE.finditer(text):
            out.append(canonicalize_tech(text[last:m.start()]))
            out.append(m.group())
            last = m.end()
        out.append(canonicalize_tech(text[last:]))
        return "".join(out)

    for pattern, canonical, ambiguous in _TECH_RULES:
        def swap(m, canonical=canonical, ambiguous=ambiguous):
            found = m.group()
            # A lowercase split of a name that starts with an ordinary English word is
            # prose, not a product: "a fast API response" must survive untouched.
            if ambiguous and re.search(r"[\s\-]", found) and not found[:1].isupper():
                return found
            return canonical
        text = pattern.sub(swap, text)
    return text


def _text(value) -> str:
    """A model's string, stripped of control characters and outer whitespace."""
    return _CONTROL_RE.sub("", str(value or "")).strip()


def _squash(text) -> str:
    """Lowercase letters and digits only, so 'U.S. Citizen' and 'US citizen' compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _strings(node):
    """Every string value in a nested json_data node (keys excluded)."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _month_year(value) -> str:
    """'2021-05' → 'May 2021', '2021' → '2021', anything else → ''."""
    m = _ISO_DATE_RE.match(str(value or "").strip())
    if not m:
        return ""
    year, month = m.group(1), m.group(2)
    if month is None:
        return year
    return f"{_MONTHS[int(month) - 1]} {year}" if 1 <= int(month) <= 12 else ""


def _role_dates(role: dict) -> str:
    """'Mon YYYY – Mon YYYY' from the base's ISO start/end, or the single date on its own.

    A résumé that prints one date for a role is stating when it ended, so that date prints
    alone: "Jun 2025", never "– Jun 2025" or "Jun 2025 –". A dangling dash reads as missing
    information to a person and leaves a parser looking for the other half of a range.
    Returns '' only when there is no date at all, so the printed `dates` string still stands.
    """
    start = _month_year(role.get("start"))
    end = "Present" if role.get("current") else _month_year(role.get("end"))
    if start and end:
        return f"{start} – {end}"
    return end or start or ""


def _graduation_date(edu: dict) -> str:
    """When an education entry ended, 'Mon YYYY' where the month is known; '' while it is ongoing or undated."""
    if edu.get("current"):
        return ""
    return _month_year(edu.get("end"))


def _contact_kind(item: dict) -> str | None:
    """The header field a printed contact item is, or None for one the header leaves out (city, portfolio, …)."""
    text, url = str(item.get("text") or ""), str(item.get("url") or "")
    both = f"{text} {url}".lower()
    if "linkedin" in both:
        return "linkedin"
    if "github" in both:
        return "github"
    if url.lower().startswith("mailto:") or _EMAIL_RE.search(text):
        return "email"
    if url.lower().startswith("tel:") or _PHONE_RE.search(text):
        return "phone"
    return None


def _tailored_header(base_header: dict, reply_header: dict, base_text: str) -> dict:
    """The name, then location, email, phone, work authorization, GitHub and LinkedIn: the base's own contact items, in that order.

    Items are copied whole (text, url, tracer stub), never rewritten from the reply.
    Location and work authorization are the two fields a parse never splits out of a
    contact line, so the reply points at each and the item printing it is kept; a value
    the base prints elsewhere (say, in the summary) becomes a plain item. Every other
    item is left out.
    """
    items = [i for i in (base_header.get("contact_items") or []) if isinstance(i, dict)]
    contact = base_header.get("contact") if isinstance(base_header.get("contact"), dict) else {}
    picked = {}
    for item in items:
        kind = _contact_kind(item)
        if kind and kind not in picked:
            picked[kind] = dict(item)

    # A field the parse split out of the header but the contact line never printed.
    for kind in ("location", "email", "phone", "github", "linkedin"):
        value = str(contact.get(kind) or "").strip()
        if value and kind not in picked:
            url = {"email": f"mailto:{value}", "github": value, "linkedin": value}.get(kind, "")
            picked[kind] = {"text": value, "url": url}

    # Neither of these has a shape a parse can recognise on its own, so the reply names
    # the value and the base's own item printing it is kept — the reply never writes one.
    for kind in ("location", "work_authorization"):
        if kind in picked:
            continue
        value = _text((reply_header or {}).get(kind))
        key = _squash(value)
        if len(key) < 3:
            continue
        for item in items:
            if _contact_kind(item) is None and key in _squash(item.get("text")):
                picked[kind] = dict(item)
                break
        else:
            if len(key) >= 5 and key in base_text:
                picked[kind] = {"text": value, "url": ""}

    # The headline may be narrowed to the part of the base's own that fits the posting
    # ("GenAI / LLM Engineer | Python Backend Engineer" -> "GenAI / LLM Engineer"), but a
    # title the base never states — a seniority, the posting's own job title — is refused.
    base_title = str(base_header.get("title") or "").strip()
    reply_title = _text((reply_header or {}).get("title"))
    title = reply_title if reply_title and _squash(reply_title) in _squash(base_title) else base_title

    header = {"name": base_header.get("name", ""),
              "title": title,
              "contact_items": [picked[k] for k in _HEADER_ORDER if k in picked]}
    if contact:
        header["contact"] = dict(contact)
    return header


def _role_key(company, title) -> tuple:
    return _squash(company), _squash(title)


def _pair_roles(base_roles: list, reply_roles: list) -> list:
    """The reply entry that rewrote each base role, or None.

    Matched on company + title, never on position alone, so a reply that reorders or
    drops a role can't hang one role's bullets under another. Position only breaks a
    tie between equal keys; a title the reply reworded still matches on a company no
    other role shares; a reply entry with a blank identity is taken on position.
    """
    used, pairs = set(), []
    base_companies = [_squash(r.get("company")) for r in base_roles]
    for i, role in enumerate(base_roles):
        key = _role_key(role.get("company"), role.get("title"))
        free = [j for j in range(len(reply_roles)) if j not in used]
        same = [j for j in free
                if _role_key(reply_roles[j].get("company"), reply_roles[j].get("job_title")) == key]
        if not same and key[0] and base_companies.count(key[0]) == 1:
            same = [j for j in free if _squash(reply_roles[j].get("company")) == key[0]][:1]
        if not same and i in free and not any(
                _role_key(reply_roles[i].get("company"), reply_roles[i].get("job_title"))):
            same = [i]
        j = i if i in same else (same[0] if same else None)
        if j is not None:
            used.add(j)
        pairs.append(reply_roles[j] if j is not None else None)
    return pairs


def _skills_map(pairs) -> dict:
    """technical_skills pairs → the {category: "a, b"} map json_data stores."""
    out = {}
    for pair in pairs if isinstance(pairs, list) else []:
        if not isinstance(pair, dict):
            continue
        label = _text(pair.get("category"))
        value = pair.get("skills")
        if isinstance(value, list):
            value = ", ".join(str(v).strip() for v in value if str(v).strip())
        value = canonicalize_tech(_text(value))
        if label and value:
            out[label] = f"{out[label]}, {value}" if label in out else value
    return out


def _initials(text: str) -> str:
    """'Azure Data Factory' → 'adf'; '' when there is nothing to initial."""
    words = re.findall(r"[A-Za-z0-9]+", str(text or ""))
    return "".join(w[0] for w in words).lower() if len(words) > 1 else ""


# A vendor's name in front of its own product does not make it a different product:
# "AWS CloudWatch" is CloudWatch, "Apache Cassandra" is Cassandra. Any OTHER extra word
# does: "GitHub Actions" is not GitHub, and "Spring Data JPA" is not JPA.
_VENDOR_PREFIXES = ("apache", "aws", "amazon", "microsoft", "azure", "google", "oracle",
                    "ibm", "redhat", "atlassian", "adobe")


_LABEL_NOISE = {"and", "amp", "other", "misc", "general"}


def _label_words(label: str) -> set:
    """The meaningful words of a skills heading, for deciding where a renamed category went."""
    return {w for w in re.findall(r"[a-z0-9]+", str(label or "").lower())
            if w not in _LABEL_NOISE}


def _split_skills(value) -> list:
    """A skills line into its items, ignoring commas inside brackets.

    "AWS (ECS, Lambda, S3), Docker" is two skills, not five. Splitting blindly on every
    comma tore "AWS (ECS" away from "Lambda" and left "S3)" carrying a stray bracket, and
    the backfill then restored those fragments as orphans on their own lines — one tailored
    copy printed "Cloud: Google Cloud Platform (GCP) (Cloud Run" and "Containers & DevOps: IaC".
    """
    items, depth, current = [], 0, []
    for ch in str(value or ""):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    items.append("".join(current).strip())
    return [i for i in items if i]


# "Google Cloud Platform (GCP)" spells its own acronym out. For PRINTING that is what the
# résumé wrote and stays; for COMPARING it has to reduce to the same key as a reply that
# wrote "GCP", or the two are treated as different skills and both get printed.
_SPELLED_ACRONYM_RE = re.compile(r"((?:\b[A-Za-z][A-Za-z0-9.+&/-]*\s+){1,5})\(([A-Za-z]{2,8})\)")


def _drop_spelled_acronym(text: str) -> str:
    def cut(m):
        words = re.findall(r"[A-Za-z0-9]+", m.group(1))
        initials = "".join(w[0] for w in words).lower()
        # Only when the bracket really is the initials of the words in front of it, and
        # the ACRONYM is what survives: a reply writing the short form has to reduce to
        # the same key, and no alias table knows every long form a résumé might spell out.
        return m.group(2) + " " if initials.endswith(m.group(2).lower()) else m.group(0)
    return _SPELLED_ACRONYM_RE.sub(cut, text)


# "AWS (EC2, EKS, S3, ...)" names one skill, AWS, and then lists what was used inside it.
# A reply that writes plain "AWS" means the same skill, so the bracketed detail comes off
# before comparing — otherwise the floor restores the long form next to the short one and
# the line reads "AWS, Microsoft Azure, GCP, AWS (EC2, ...), Microsoft Azure (AKS, ...)".
_DETAIL_GROUP_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _skill_key(skill: str) -> str:
    """One key per skill: used only for comparing, never for what gets printed."""
    named = canonicalize_tech(_drop_spelled_acronym(str(skill or "")))
    trimmed = _DETAIL_GROUP_RE.sub("", named).strip()
    return _squash(trimmed or named)


def _same_skill(a: str, b: str) -> bool:
    """Whether two spellings name the same skill.

    Canonicalise first, then compare exactly. This used to test whether one name contained
    the other, which quietly merged every skill whose name sits inside another: "SQL" is
    inside "PostgreSQL", "MySQL" and "SQL Server", so a résumé listing SQL lost all three —
    the same four terms on every run, whatever the posting asked for. Containment was a
    crude stand-in for "these are the same product"; canonicalize_tech() does that properly,
    from a curated list, so "Postgres" and "Postgres SQL" still meet "PostgreSQL" here.
    """
    ka, kb = _skill_key(a), _skill_key(b)
    if not ka or not kb:
        return False
    if ka == kb or ka == kb + "s" or kb == ka + "s":   # VLAN / VLANs, API / APIs
        return True
    return _drop_vendor(ka) == _drop_vendor(kb) and _drop_vendor(ka) != ""


def _drop_vendor(key: str) -> str:
    for vendor in _VENDOR_PREFIXES:
        if key.startswith(vendor) and len(key) > len(vendor) + 2:
            return key[len(vendor):]
    return key


def _skill_is_grounded(skill: str, base_text: str, base_terms: list) -> bool:
    """Whether the base states this skill, allowing for the JD's wording of it.

    Direct either way, so the reply may expand ('Cassandra' → 'Apache Cassandra') or
    contract ('Retrieval-Augmented Generation (RAG)' → 'RAG'). Rule 5 also asks the reply
    to spell a skill the way the posting does, so an acronym the base states is matched
    against the initials of the reply's spelling: base 'ADF' keeps the reply's
    'Azure Data Factory' rather than reverting it.
    """
    key = _squash(skill)
    if not key:
        return False
    # A one- or two-letter name ("C", "R", "Go") is a substring of everything, so it has to
    # be matched against the base's own terms rather than against the résumé's whole text.
    if len(key) <= 2:
        return any(_squash(t) == key for t in base_terms)
    if key in base_text:
        return True
    if _squash(_initials(skill)) in {_squash(t) for t in base_terms if len(_squash(t)) >= 2}:
        return True
    return any(len(_squash(t)) > 2 and _squash(t) in key for t in base_terms)


def _merge_skills(reply_skills: dict, base_skills: dict, base_text: str = "") -> dict:
    """The reply's skills, with every base skill it left out put back and every one it invented taken out.

    The reply chooses the grouping, the order and each skill's wording; the base decides
    membership, in both directions. A model asked to tailor a skills list compresses it —
    one reply turned 11 categories into 5 and lost 18 terms, DHCP and SSH among them — and
    a skill dropped here is keyword surface gone, so omission cannot mean "leave it out". The
    same reply added "TCP/IP" and "network topology" because the posting asked for them,
    which the résumé states nowhere; the base is the ceiling as well as the floor. A base
    skill counts as already present when its squashed form contains, or is contained by, one
    the reply states, so "VLAN" is not re-added beside the reply's "VLANs".
    """
    if not reply_skills:
        return dict(base_skills or {})

    def split(value):
        return _split_skills(value)

    base_terms = [s for value in (base_skills or {}).values() for s in split(value)]
    if base_text:
        reply_skills = {label: ", ".join(kept_terms)
                        for label, value in reply_skills.items()
                        if (kept_terms := [s for s in split(value)
                                           if _skill_is_grounded(s, base_text, base_terms)])}
        if not reply_skills:
            return dict(base_skills or {})

    # The reply may name a skill more briefly than the résumé does ("AWS" for
    # "AWS (EC2, EKS, S3, ...)"). Same skill, so it is not restored twice — and the fuller
    # spelling is the one printed, because the detail is what a posting matches on.
    richer = {}
    for term in base_terms:
        for value in list(reply_skills):
            for short in split(reply_skills[value]):
                if _same_skill(short, term) and len(term) > len(short):
                    richer[short] = term
    if richer:
        reply_skills = {label: ", ".join(richer.get(s, s) for s in split(value))
                        for label, value in reply_skills.items()}

    # Substituting the fuller spelling can land the same skill twice on one line, when the
    # reply named it both ways ("AWS, ... , AWS (EKS, ECS, EC2)"). First mention wins.
    def dedupe(items):
        out = []
        for item in items:
            if not any(_same_skill(item, seen) for seen in out):
                out.append(item)
        return out

    reply_skills = {label: ", ".join(dedupe(split(value)))
                    for label, value in reply_skills.items()}

    kept = [s for value in reply_skills.values() for s in split(value)]

    def present(skill: str) -> bool:
        return any(_same_skill(skill, k) for k in kept)

    out = {label: value for label, value in reply_skills.items()}
    by_squashed = {_squash(label): label for label in out}
    for label, value in (base_skills or {}).items():
        # A base category holding credentials is the one drop to respect: certifications
        # have their own section, and _certifications() reads this category to fill it.
        if "certif" in str(label).lower():
            continue
        missing = [s for s in split(value) if not present(s)]
        if not missing:
            continue
        # Back into the reply's own category when it kept one by that name; otherwise
        # into whichever reply category already took this one's siblings, because the
        # reply is allowed to rename and merge categories. Re-creating the old heading
        # for a leftover or two printed thin orphan lines ("Containers & DevOps: IaC")
        # next to the merged line that had swallowed everything else.
        target = by_squashed.get(_squash(label))
        if not target:
            # Matched on the heading, not on its contents: "Cloud" belongs with
            # "Cloud & DevOps", and "Containers & DevOps" with it too, while "Tools &
            # Methodologies" belongs with neither. Counting shared skills instead swept
            # fourteen tools under a cloud heading because both lines mentioned CI/CD.
            mine = _label_words(label)
            kin = [(len(mine & _label_words(rl)), rl) for rl in reply_skills]
            best, label_of_best = max(kin, default=(0, None))
            target = label_of_best if best else None
        if target:
            out[target] = f"{out[target]}, {', '.join(missing)}"
        else:
            out[label] = ", ".join(missing)
            by_squashed[_squash(label)] = label
    return out


# Tokens that read as a claim about technology rather than ordinary prose: acronyms
# (DHCP, SSH, ERP), names carrying a digit (802.1Q, IPv6, Microsoft 365, GPT-4o) and
# CamelCase products (LangGraph, FastAPI). Ordinary words are left alone — the skills
# vocabulary of the base résumé catches the lowercase ones ("logging", "interfaces").
_CLAIM_RES = (
    re.compile(r"\b[A-Z][A-Za-z]*[0-9][\w.\-]*\b"),
    re.compile(r"\b[A-Z]{2,8}\b"),
    re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]*)+\b"),
)


def unsupported_claims(tailored: dict, base: dict) -> list:
    """Technical terms a tailored copy states that its base does not support, as (section, term) pairs.

    Rule 0 made checkable: a bullet may name a tool or technique only if the role it sits
    under already names it — the skills list and the other roles do not count — and the
    summary may only say what the résumé says. Reporting only, never editing: a rewrite
    that merges two bullets is legitimate and a dropped bullet costs more than a flagged
    one, so this marks work for a human rather than deciding for them.
    """
    base = base or {}
    vocabulary = [s.strip() for value in (base.get("skills") or {}).values()
                  for s in _split_skills(value) if len(s.strip()) > 2]

    def terms(text: str) -> set:
        found = {m for pattern in _CLAIM_RES for m in pattern.findall(text or "")}
        found |= {v for v in vocabulary if _squash(v) and _squash(v) in _squash(text)}
        return found

    def unsupported(text: str, allowed: str) -> list:
        allowed = _squash(allowed)
        return sorted({t for t in terms(text) if _squash(t) not in allowed})

    out = []
    for term in unsupported(tailored.get("summary"), json.dumps(base, ensure_ascii=False)):
        out.append(("summary", term))

    base_roles = {_squash(r.get("company")): r for r in (base.get("experience") or [])
                  if isinstance(r, dict)}
    for role in tailored.get("experience") or []:
        if not isinstance(role, dict):
            continue
        source = base_roles.get(_squash(role.get("company")))
        if source is None:
            continue
        allowed = json.dumps(source, ensure_ascii=False)
        for bullet in role.get("bullets") or []:
            for term in unsupported(bullet, allowed):
                out.append((f"experience/{role.get('company')}", term))
    return out


def dropped_source_terms(structured: dict, source_text: str) -> list:
    """Technical terms the source document names that the structured résumé no longer holds.

    The structuring step has no pool to fall back on — the document is the pool — so a term
    lost here is lost for good, and tailoring can never put it back. It happens when a skills
    line is written as a sentence and only its subject survives: "Ping, Traceroute, and
    Nslookup for connectivity, latency and DNS diagnostics" came through without DNS, a term
    the posting asked for by name. Reporting only, so a human sees what the document gave up.
    """
    kept = _squash(json.dumps(structured, ensure_ascii=False))
    out, seen = [], set()
    for line in str(source_text or "").splitlines():
        letters = [c for c in line if c.isalpha()]
        # A heading is set in capitals, so its words would read as acronyms.
        if not letters or sum(c.isupper() for c in letters) / len(letters) > 0.9:
            continue
        for pattern in _CLAIM_RES:
            for term in pattern.findall(line):
                key = _squash(term)
                if len(key) >= 2 and key not in kept and key not in seen:
                    seen.add(key)
                    out.append(term)
    return out


def _certifications(reply_certs, base: dict) -> list:
    """The reply's certifications that the base itself states, deduplicated, in the reply's order.

    Checked against where a base can state one (its own list, a skills category named
    for them, a line that mentions a certification) rather than the whole résumé, so a
    skill can't come back relabelled as a credential.
    """
    sources = list(_strings(base.get("certifications")))
    skills = base.get("skills")
    if isinstance(skills, dict):
        sources += [s for label, value in skills.items() if "certif" in str(label).lower()
                    for s in _strings(value)]
    sources += [s for s in _strings(base) if "certif" in s.lower()]
    pool = _squash(" ".join(sources))
    out, seen = [], set()
    for cert in reply_certs if isinstance(reply_certs, list) else []:
        text = canonicalize_tech(_text(cert))
        key = _squash(text)
        if len(key) >= 3 and key in pool and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _clean_bullets(bullets) -> list:
    return [t for t in (canonicalize_tech(_text(b))
                        for b in (bullets if isinstance(bullets, list) else [])) if t]


def _profile_link(value: str, domain: str) -> dict | None:
    """A clickable contact item for a GitHub/LinkedIn address, or None when the value is no address (a bare "LinkedIn" label can't be made clickable)."""
    value = value.strip()
    if domain not in value.lower():
        return None
    url = value if re.match(r"^https?://", value, re.IGNORECASE) else f"https://{value}"
    text = re.sub(r"^https?://(www\.)?", "", value, flags=re.IGNORECASE).rstrip("/")
    return {"text": text, "url": url}


def _project_url(value: str) -> str:
    """A project link fit to click, or '' for a value that is no address."""
    value = value.strip()
    return value if re.match(r"^(https?://)?[\w-]+(\.[\w-]+)+(/\S*)?$", value, re.IGNORECASE) else ""



def build_base_resume(reply: dict) -> dict:
    """A structuring reply → a base résumé's json_data: only the ATS sections, in order, marked for the ATS layout.

    Each role's printed date is formatted from its ISO pair ("Mon YYYY – Mon YYYY") and
    GitHub and LinkedIn become clickable only when the reply holds a real address. No
    bullet is trimmed here: the base résumé is the pool tailoring draws from, so it keeps
    everything the document states.
    """
    reply = reply if isinstance(reply, dict) else {}
    head = reply.get("header") if isinstance(reply.get("header"), dict) else {}

    def field(key):
        return _text(head.get(key))

    # The value alone: a label the model kept ("Email: …") is not part of it.
    email = (_EMAIL_RE.search(field("email")) or [""])[0]
    phone_match = _PHONE_RE.search(field("phone"))
    phone = phone_match.group(0).strip() if phone_match else field("phone")
    github = _profile_link(field("github"), "github.")
    linkedin = _profile_link(field("linkedin"), "linkedin.")
    location = field("location")
    items = []
    # Location leads the contact line, as a résumé prints it: it is what a job board's
    # geographic filter reads, and the header is the only place it looks.
    if location:
        items.append({"text": location, "url": ""})
    if email:
        items.append({"text": email, "url": f"mailto:{email}"})
    if phone:
        items.append({"text": phone, "url": ""})
    if field("work_authorization"):
        items.append({"text": field("work_authorization"), "url": ""})
    items += [link for link in (github, linkedin) if link]
    header = {
        "name": field("name"),
        # The headline the résumé prints under the name; the templates leave the line out when empty.
        "title": field("title"),
        "contact_items": items,
        # The split-out fields tailoring falls back on when the contact line lacks one.
        "contact": {"location": location, "email": email, "phone": phone,
                    "github": github["url"] if github else "",
                    "linkedin": linkedin["url"] if linkedin else ""},
    }

    experience = []
    for r in reply.get("work_experience") or []:
        if not isinstance(r, dict):
            continue
        role = {
            "company": _text(r.get("company")),
            "title": _text(r.get("job_title")),
            "location": _text(r.get("location")),
            "date": "",
            "start": str(r.get("start") or "").strip(),
            "end": str(r.get("end") or "").strip(),
            "current": bool(r.get("current")),
            "description": "",
            "bullets": _clean_bullets(r.get("bullets")),
        }
        role["date"] = _role_dates(role) or _text(r.get("dates"))
        experience.append(role)

    projects = []
    for p in reply.get("projects") or []:
        if not isinstance(p, dict):
            continue
        # The printed dates and stack ride in the description slot, one line under the name.
        subtitle = " | ".join(s for s in (_text(p.get("dates")),
                                          canonicalize_tech(_text(p.get("technologies")))) if s)
        projects.append({
            "name": _text(p.get("name")),
            "url": _project_url(_text(p.get("url"))),
            "description": subtitle,
            "bullets": _clean_bullets(p.get("bullets")),
        })

    # An entry with no bullets is not experience: models have answered with a second,
    # bullet-less copy of a role, or parked the employer's description in one.
    experience = [r for r in experience if r["bullets"]]
    projects = [p for p in projects if p["bullets"]]

    education = [
        {"school": _text(e.get("institution")),
         "degree": _text(e.get("degree")),
         # The date as the document prints it ("Jun 2025", "Expected Dec 2026"), which
         # the templates put on the degree line rather than in a right-hand column.
         "years": _text(e.get("graduation_date")),
         "grade": _text(e.get("grade"))}
        for e in reply.get("education") or [] if isinstance(e, dict)
    ]

    certifications, seen = [], set()
    for cert in reply.get("certifications") or []:
        text = canonicalize_tech(_text(cert))
        if text and _squash(text) not in seen:
            seen.add(_squash(text))
            certifications.append(text)

    return drop_empty_entries({
        "header": header,
        "summary": canonicalize_tech(_text(reply.get("professional_summary"))),
        "skills": _skills_map(reply.get("technical_skills")),
        "experience": experience,
        "projects": projects,
        "education": education,
        "certifications": certifications,
        LAYOUT_KEY: ATS_LAYOUT,
    })


def _tailored_summary(reply: dict, base: dict, jd_text: str) -> str:
    """The reply's summary, unless rewriting it dropped something the posting asked for.

    A rewritten summary is usually the better one — it leads with what the posting wants.
    But the model shortens as it rewrites, and a term this posting names is the last thing
    that should fall out of the first paragraph a reader and a keyword matcher both read.
    Keeping the base's summary loses the rewrite; losing the keyword loses the match.
    """
    written = canonicalize_tech(_text(reply.get("professional_summary")))
    original = base.get("summary", "")
    if not written:
        return original
    if not original or not jd_text:
        return written
    kept = _squash(canonicalize_tech(written))
    was = _squash(canonicalize_tech(original))
    lost = [term for term in jd_protected_terms(base, jd_text)
            if _skill_key(term) in was and _skill_key(term) not in kept]
    if lost:
        logger.warning("Tailor: keeping the base summary — the rewrite dropped %s, "
                       "which this posting asks for", ", ".join(lost[:8]))
        return original
    return written


def jd_protected_terms(base: dict, jd_text: str) -> list:
    """Technology names the base résumé states that this posting also names.

    These are the terms the whole tailoring run exists to surface, so they are the one
    thing trimming may never remove. Measured before this existed: the summary lost about a
    third of its words on every run, taking PostgreSQL, Kafka, Terraform and CI/CD with it
    on a posting that named all four.
    """
    jd = _squash(jd_text)
    if not jd:
        return []
    out, seen = [], set()
    for value in (base.get("skills") or {}).values():
        for term in _split_skills(value):
            term = term.strip()
            key = _skill_key(term)
            # Two characters or fewer match far too much of any prose to test this way.
            if len(key) > 2 and key in jd and key not in seen:
                seen.add(key)
                out.append(term)
    return out


def _resume_vocabulary(base: dict) -> set:
    """Every technology name THIS résumé uses, taken from its own skills section.

    Nothing about technology is written down here. A résumé that lists PyTorch and YOLO
    gets those; one that lists GKE and Terraform gets those. The set is only ever used to
    decide which words in a bullet are worth policing, so a résumé the code has never
    seen carries its own vocabulary in with it.
    """
    vocab = set()
    for value in (base.get("skills") or {}).values():
        for term in _split_skills(str(value or "")):
            bare = _DETAIL_GROUP_RE.sub("", term).strip()
            for piece in (term, bare):
                key = _squash(piece)
                if len(key) >= 2:
                    vocab.add(key)
                # the résumé's bullets may use the acronym where its skills spell the name
                # out ("ADF" for "Azure Data Factory"), so the short form counts too
                short = _squash(_initials(piece))
                if len(short) >= 2:
                    vocab.add(short)
            # a group like "GCP (GKE, Compute Engine)" also contributes its members
            inner = re.search(r"\(([^()]*)\)\s*$", term)
            if inner:
                for member in inner.group(1).split(","):
                    key = _squash(member)
                    if len(key) >= 2:
                        vocab.add(key)
    return vocab


def _named_tech(text: str, vocab: set) -> set:
    """The vocabulary terms this sentence names, as written, matched on whole words."""
    words = re.findall(r"[A-Za-z0-9+#./-]+", str(text or ""))
    found = set()
    for n in (3, 2, 1):                       # longest phrase first: "AWS Secrets Manager"
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            if _squash(phrase) in vocab:
                found.add(phrase)
    return found


def _same_tech_mention(a: str, b: str) -> bool:
    """Whether two mentions name the same tool, allowing the spelling rule 5 asks for.

    Rule 5 tells the reply to spell a tool the way the posting does, so a bullet that says
    "ADF" comes back as "Azure Data Factory" — the same claim, expanded. `_skill_is_grounded`
    already solves this for the skills list with `_initials`; a bullet needs the same
    latitude or a legitimate expansion reads as a new tool.
    """
    if _same_skill(a, b):
        return True
    ka, kb = _squash(a), _squash(b)
    return (_squash(_initials(a)) == kb and kb) or (_squash(_initials(b)) == ka and ka)


def _ground_bullets(rewritten: list, base_bullets: list, vocab: set) -> list:
    """Each rewritten bullet, unless it names a technology its own source bullet did not.

    Rule 0 already says a bullet may name a tool only if the bullet it rewrites, or the
    role it sits under, already named it. This is that rule as a check. The failure is
    real and reproducible: on a GCP posting the model rewrote "aligning Kubernetes
    operations" to "aligning GKE operations" in every run measured — narrowing a true
    statement about three clouds into a claim about Google's product, which the résumé
    never made. Two prompt rules and a change of model each failed to stop it.

    A bullet that drifts is replaced by the base's own sentence, so the worst this can do
    is print what the résumé already said. It knows no technology names of its own: the
    vocabulary arrives from the résumé being tailored, and a term the role already uses
    anywhere is allowed, so a genuine rewrite that moves a tool between a role's bullets
    is untouched.
    """
    if not rewritten or not base_bullets:
        return rewritten
    role_terms = set()
    for b in base_bullets:
        role_terms |= _named_tech(b, vocab)

    out = []
    for bullet in rewritten:
        named = _named_tech(bullet, vocab)
        extra = {t for t in named
                 if not any(_same_tech_mention(t, r) for r in role_terms)}
        if not extra:
            out.append(bullet)
            continue
        source = difflib.get_close_matches(bullet, base_bullets, 1, 0.45)
        if source:
            logger.warning("Tailor: bullet named %s, which its role never states — keeping the "
                           "résumé's own sentence", ", ".join(sorted(extra)))
            out.append(source[0])
        else:
            # Nothing close enough to revert to; keep it and let unsupported_claims flag it.
            logger.warning("Tailor: bullet named %s with no source bullet to fall back on: %.90s",
                           ", ".join(sorted(extra)), bullet)
            out.append(bullet)
    return out


def build_tailored_resume(reply: dict, base: dict, jd_text: str = "") -> dict:
    """A tailor reply folded onto its base: json_data holding only the ATS sections, in order.

    The reply supplies the prose (summary, technical skills, each role's and project's
    bullets) and picks the projects and certifications. Facts come from the base, never
    the reply: the header's items, each role's company, title, location and dates, each
    project's name, link, dates and stack, and education. A section the reply left empty
    keeps the base's, and a role no reply entry matches keeps its base bullets; a project the reply
    omits keeps its base bullets, and one the base doesn't hold is never added. Skills the
    reply leaves out come back from the base: it groups, orders and words them, the base
    decides which exist. Publications are not carried over.
    """
    reply = reply if isinstance(reply, dict) else {}
    base = base or {}
    base_text = _squash(json.dumps(base, ensure_ascii=False))

    _vocab = _resume_vocabulary(base)

    base_roles = [r for r in (base.get("experience") or []) if isinstance(r, dict)]
    reply_roles = [r for r in (reply.get("work_experience") or []) if isinstance(r, dict)]
    experience = []
    for role, rewritten in zip(base_roles, _pair_roles(base_roles, reply_roles)):
        role = dict(role)
        role.pop("suggested_bullets", None)
        if rewritten is not None and isinstance(rewritten.get("bullets"), list):
            role["bullets"] = _ground_bullets(_clean_bullets(rewritten["bullets"]),
                                              role.get("bullets") or [], _vocab)
            # The prompt turns a role's paragraph into bullets: bullets, never paragraphs.
            role["description"] = ""
        dates = _role_dates(role)
        if dates:
            role["date"] = dates
        experience.append(role)

    base_projects = [p for p in (base.get("projects") or []) if isinstance(p, dict)]
    if "projects" not in reply:
        # A reply that never addressed projects (a provider without the schema) drops none.
        projects = [dict(p) for p in base_projects]
    else:
        # The reply names the projects worth keeping; each keeps its base facts, in the
        # base's order, and only a project the base holds can come back.
        rewritten = {}
        for rp in reply.get("projects") or []:
            if isinstance(rp, dict) and _squash(rp.get("name")) not in rewritten:
                rewritten[_squash(rp.get("name"))] = _clean_bullets(rp.get("bullets"))
        projects = []
        for p in base_projects:
            key = _squash(p.get("name"))
            match = rewritten.get(key)
            if match is None:
                match = next((b for k, b in rewritten.items()
                              if len(k) >= 8 and (key.startswith(k) or k.startswith(key))), None)
            # A project the reply never mentions keeps its base bullets: models drop one by
            # omission often enough that silence cannot mean "leave it out". Dropping one
            # takes saying so — returning it with no bullets at all.
            if match is None:
                projects.append(dict(p))
            elif match:
                projects.append({**p, "bullets": match})

    education = []
    for edu in base.get("education") or []:
        if not isinstance(edu, dict):
            continue
        edu = dict(edu)
        date = _graduation_date(edu)
        if date:
            edu["years"] = date
        education.append(edu)

    base_header = base.get("header") if isinstance(base.get("header"), dict) else {}
    reply_header = reply.get("header") if isinstance(reply.get("header"), dict) else {}
    return {
        "header": _tailored_header(base_header, reply_header, base_text),
        "summary": _tailored_summary(reply, base, jd_text),
        "skills": _merge_skills(_skills_map(reply.get("technical_skills")),
                                base.get("skills") or {}, base_text),
        "experience": experience,
        "projects": projects,
        "education": education,
        "certifications": _certifications(reply.get("certifications"), base),
        LAYOUT_KEY: ATS_LAYOUT,
    }
