"""
Simple two-pane data-annotation tool
------------------------------------

• Left column shows the raw JSON (system/user/input prompt etc.)
• Right-top shows the model output
• Right-bottom is a textarea for the reviewer’s note

Data is loaded from a single JSONL file (see DATA_PATH below).
Annotations are written to `data/annotation/<file>.annotations.jsonl`

Run with:
    python src/annotator.py

The app is intentionally kept in one file for quick experimentation.
"""

import json
import re
import textwrap
from pathlib import Path

from fasthtml.common import *
from starlette.requests import Request
from starlette.responses import RedirectResponse

###############################################################################
# Task parsing and rendering helpers
###############################################################################


def _parse_tasks(text: str) -> list[dict]:
    """
    Extract task dictionaries from a block of text delimited by
    TASK_START … TASK_END, where each line within a block is `KEY: value`.
    Returns an empty list when no tasks are found.
    """
    blocks = re.findall(r"TASK_START(.*?)TASK_END", text, flags=re.DOTALL)
    tasks: list[dict] = []
    for blk in blocks:
        task: dict[str, str] = {}
        for line in blk.strip().splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            task[k.strip().lower()] = v.strip()
        if task:
            tasks.append(task)
    return tasks


def TaskCard(task: dict):
    "Very small visual card for a task dict produced by `_parse_tasks`."
    title = task.get("title", "Untitled")
    assignee = task.get("assignee", "unassigned")
    prio = task.get("priority", "—")
    desc = task.get("description", "")
    meta = f"{assignee} • {prio}"
    return Div(
        H4(title, cls="task-title"),
        P(meta, cls="task-meta"),
        P(desc, cls="task-desc"),
        cls="task-card",
    )


def _parse_summary(text: str) -> dict[str, str] | None:
    """
    Very lightweight parser for summary blocks that follow the
    TITLE/SUMMARY/KEY_POINTS/PARTICIPANTS/TOPICS convention.
    Returns a dict with those keys lower-cased; returns None if TITLE or
    SUMMARY missing.
    """
    # Make extraction robust to extra whitespace and missing sections
    sections = {}
    current_key = None
    lines = text.splitlines()
    for ln in lines:
        # new section?
        m = re.match(r"^([A-Z_]+)\s*:\s*(.*)", ln.strip())
        if m:
            current_key, first_val = m.groups()
            current_key = current_key.lower()
            sections[current_key] = first_val.strip()
        elif current_key:
            # continuation of previous section
            sections[current_key] += (
                "\n" if sections[current_key] else ""
            ) + ln.rstrip()
    # Normalise KEY_POINTS bullets → list of str
    if "key_points" in sections:
        bullets = []
        for l in sections["key_points"].splitlines():
            l = l.strip()
            if l.startswith(("*", "-")):
                bullets.append(l.lstrip("*- ").rstrip())
            elif l:
                bullets.append(l)
        sections["key_points"] = bullets
    # Require at minimum title + summary
    return sections if {"title", "summary"} <= sections.keys() else None


def SummaryCard(summary: dict):
    "Tiny component to display a parsed summary dictionary."
    bullets = summary.get("key_points", [])
    return Div(
        H4(summary.get("title", "Untitled"), cls="summary-title"),
        P(summary.get("summary", ""), cls="summary-body"),
        (Ul(*[Li(b) for b in bullets]) if bullets else ""),
        P(
            "Participants: " + summary.get("participants", "—"),
            cls="summary-meta",
        )
        if summary.get("participants")
        else "",
        P("Topics: " + summary.get("topics", "—"), cls="summary-meta")
        if summary.get("topics")
        else "",
        cls="summary-card",
    )


###############################################################################
# Configuration
###############################################################################

# Point this to *your* dataset. Using the user’s example path by default.
DATA_PATH = Path(
    "/Users/strickvl/coding/zenml/repos/zenml-projects/llm-daily-summarization/data/private_data/1753454885402-lf-observations-export-cmdeblxja02dkad071lfuqsgx.jsonl"
)
ANNOT_DIR = Path(
    "/Users/strickvl/coding/zenml/repos/zenml-projects/llm-daily-summarization/data/annotation"
)

# The annotation file will be created automatically beside DATA_PATH.
ANNOT_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_PATH = ANNOT_DIR / f"{DATA_PATH.stem}.annotations.jsonl"

###############################################################################
# Data helpers
###############################################################################


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    try:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data file {path}: {e}")
        return []


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    """Write list of dicts back to JSONL, *overwriting* existing file."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pretty_json(obj: dict | list) -> str:
    """Return indented JSON for nicer display in the left pane."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


###############################################################################
# Persistence layer for annotations
###############################################################################


class AnnotationStore:
    """Lightweight in-memory cache synchronised with a JSONL file."""

    def __init__(self, source_count: int, path: Path) -> None:
        self.path = path
        # Pre-populate with empty notes
        self._data: dict[int, str] = {i: "" for i in range(source_count)}
        if path.exists():
            for line in load_jsonl(path):
                # The 'index' key was being saved as a string in some versions
                idx = int(line.get("index", -1))
                if idx != -1:
                    self._data[idx] = str(line.get("note", ""))

    def get(self, idx: int) -> str:
        return self._data.get(idx, "")

    def set(self, idx: int, note: str) -> None:
        self._data[idx] = note

    def save(self) -> None:
        rows = [
            {"index": idx, "note": note}
            for idx, note in sorted(self._data.items())
            if note.strip()  # only persist non-empty notes
        ]
        dump_jsonl(self.path, rows)


###############################################################################
# FastHTML app & routes
###############################################################################

app, rt = fast_app()
SOURCE_DATA: list[dict] = load_jsonl(DATA_PATH)
store = AnnotationStore(len(SOURCE_DATA), ANNOT_PATH)


@rt("/")
def root():
    """Redirect to the first item."""
    return RedirectResponse("/item/0", status_code=303)


@rt("/item/{idx}")
def show_item(request: Request, idx: int):
    """Render one data row for annotation."""
    idx = max(0, min(idx, len(SOURCE_DATA) - 1))

    if not SOURCE_DATA:
        return Title("Annotator"), Main(
            P("No data found or file could not be loaded."), cls="container"
        )

    item = SOURCE_DATA[idx]

    # --- New chat rendering logic ---
    raw_input = item.get("input", item)
    # Check if input is a dict and has a 'messages' key which is a list
    messages = (
        raw_input.get("messages") if isinstance(raw_input, dict) else None
    )

    if messages and isinstance(messages, list):
        chat_components = []
        for msg in messages:
            role = msg.get("role", "unknown").lower()
            content = str(msg.get("content", ""))
            chat_components.append(
                Div(
                    Div(role, cls="chat-message-role"),
                    # Use Pre to preserve whitespace and formatting in content
                    Pre(content, cls="chat-message-content"),
                    cls=f"chat-message role-{role}",
                )
            )
        left_pane = Div(*chat_components, cls="chat-container")
    else:
        # Fallback for old format or unexpected data
        left_pane = Pre(pretty_json(raw_input))
    # --- End new logic ---

    output_block = item.get("output", {})
    right_top_raw = output_block.get("content", pretty_json(output_block))

    # Detect if this item represents task-extraction output
    is_task_item = (
        item.get("name") == "daily_digest" or "TASK_START" in right_top_raw
    )

    # Detect summaries (tags OR obvious section markers)
    tags = item.get("metadata", {}).get("tags", [])
    is_summary_item = ("summarizer_agent" in tags) or (
        "TITLE:" in right_top_raw
        and "SUMMARY:" in right_top_raw
        and "KEY_POINTS:" in right_top_raw
    )

    if is_task_item:
        tasks = _parse_tasks(right_top_raw)
        if tasks:
            right_top_component = Div(
                *[TaskCard(t) for t in tasks], cls="task-list"
            )
        else:
            right_top_component = Pre(right_top_raw, cls="model-output")
    elif is_summary_item:
        summary = _parse_summary(right_top_raw)
        if summary:
            right_top_component = SummaryCard(summary)
        else:
            right_top_component = Pre(right_top_raw, cls="model-output")
    else:
        right_top_component = Pre(right_top_raw, cls="model-output")

    current_note = store.get(idx)

    # --- Build HTML using FT components ---
    css = Style("""
        body { margin:0; font-family: system-ui, sans-serif; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; height: 100vh; }
        .left  { padding: 1rem; overflow: auto; background:#fafafa; border-right:1px solid #ddd; }
        .right { display:flex; flex-direction:column; height:100%; }
        /* Right-top panel that shows model output */
        .output {
            flex: 1;                /* grow to fill remaining height */
            overflow-y: auto;       /* allow vertical scroll */
            overflow-x: hidden;     /* cut horizontal overflow */
            padding: 0.25rem 1rem 1rem;
        }
        /* Preserve spacing but wrap long tokens/words to avoid side-scroll */
        .model-output {
            white-space: pre-wrap;
            word-break: break-word;
            font-family: monospace;
        }
        .note   { flex: 0 0 auto; border-bottom:1px solid #ddd; padding:0; display:flex; flex-direction:column; }
        textarea {
            flex:1;
            width:100%;
            border:none;
            resize:none;
            padding:1rem;
            font-size:14px;
            font-family:inherit;
            margin:0;           /* eliminate default bottom margin that caused extra gap */
        }
        .controls { display:flex; border-top:1px solid #eee; }
        button { flex:1; padding:0.5rem; border:none; background:#007bff; color:white; cursor:pointer; }
        button:hover { background:#0069d9; }
        #save-status { padding: 0.5rem; color: green; text-align: center; }

        /* Chat display styles */
        .chat-container { display: flex; flex-direction: column; gap: 0.75rem; }
        .chat-message { border-radius: 8px; padding: 0.75rem 1rem; }
        .chat-message-role { font-weight: bold; text-transform: capitalize; margin-bottom: 0.5rem; color: #555; }
        .chat-message-content { white-space: pre-wrap; word-wrap: break-word; margin: 0; }
        .role-user { background-color: #e1f5fe; border: 1px solid #b3e5fc; }
        .role-assistant { background-color: #e8f5e9; border: 1px solid #c8e6c9; }
        .role-system { background-color: #f3e5f5; border: 1px solid #e1bee7; }
        .role-unknown { background-color: #f5f5f5; border: 1px solid #e0e0e0; }

        /* Minimal task-card styling */
        .task-card { border:1px solid #ddd; border-radius:6px; padding:0.75rem 1rem; background:#fff; margin-bottom:0.75rem; }
        .task-title { margin:0 0 0.25rem 0; }
        .task-meta { font-size:0.85rem; color:#555; margin:0 0 0.5rem 0; }
        .task-desc { margin:0; white-space:pre-wrap; }

        /* Minimal summary-card styling */
        .summary-card { border:1px solid #ddd; border-radius:6px; padding:0.75rem 1rem; background:#fff; margin-bottom:0.75rem;}
        .summary-title { margin:0 0 0.5rem 0; }
        .summary-body { margin:0 0 0.5rem 0; white-space:pre-wrap; }
        .summary-meta { font-size:0.85rem; color:#555; margin:0.25rem 0 0 0; }

        /* Top controls layout & sizing */
        .controls { display:flex; gap:0.25rem; }
        .controls button {                     /* 50 % smaller buttons */
            padding:0.25rem 0.5rem;
            font-size:0.8rem;
        }
        .top-controls { padding:0.5rem 1rem 0.25rem 1rem; border-bottom:1px solid #eee; }
    """)

    save_shortcut_script = Script("""
        function focusNote() {
            const t = document.querySelector('textarea[name=note]');
            if (t) t.focus();
        }

        document.addEventListener("keydown", function(e) {
            const meta = (e.metaKey || e.ctrlKey);
            if (meta && e.key === "Enter") {
                e.preventDefault();
                document.getElementById("saveBtn")?.click();
            } else if (meta && e.key === "ArrowRight") {
                e.preventDefault();
                document.getElementById("nextBtn")?.click();
            } else if (meta && e.key === "ArrowLeft") {
                e.preventDefault();
                document.getElementById("prevBtn")?.click();
            }
        });

        // Initial focus when the document loads
        document.addEventListener("DOMContentLoaded", focusNote);
        // Focus again after HTMX swaps the body
        document.addEventListener("htmx:load", focusNote);
    """)

    head_content = (
        Title(f"Annotator ({idx+1}/{len(SOURCE_DATA)})"),
        css,
        Script(src="https://unpkg.com/htmx.org@1.9.10"),
        save_shortcut_script,
    )

    left_panel = Div(H3(f"Input (#{idx})"), left_pane, cls="left")

    # Create navigation URLs directly
    prev_idx = max(0, idx - 1)
    next_idx = min(len(SOURCE_DATA) - 1, idx + 1)

    # --- Top-right static controls -----------------------------------------
    top_controls = Div(
        Button(
            "⬅︎ Back",
            id="prevBtn",
            hx_get=f"/item/{prev_idx}",
            hx_target="body",
            hx_swap="outerHTML",
            hx_include="#noteForm",  # include form data for save
            onclick=f"htmx.ajax('POST', '/save', {{target:'#save-status', swap:'innerHTML', values:{{idx:'{idx}', note:document.querySelector('[name=note]').value}}}})",
        ),
        Button(
            "Save (⌘+Enter)",
            id="saveBtn",
            hx_post="/save",
            hx_include="#noteForm",
            hx_target="#save-status",
            hx_swap="innerHTML",
        ),
        Button(
            "Next ➡︎",
            id="nextBtn",
            hx_get=f"/item/{next_idx}",
            hx_target="body",
            hx_swap="outerHTML",
            hx_include="#noteForm",
            onclick=f"htmx.ajax('POST', '/save', {{target:'#save-status', swap:'innerHTML', values:{{idx:'{idx}', note:document.querySelector('[name=note]').value}}}})",
        ),
        cls="controls top-controls",
    )
    # -----------------------------------------------------------------------

    right_panel = Div(
        top_controls,  # fixed nav buttons
        Div(  # annotation section now immediately below buttons
            Form(
                Textarea(
                    current_note,
                    name="note",
                    placeholder="Write your notes here…",
                    autofocus=True,
                ),
                Input(type="hidden", name="idx", value=idx),
                id="noteForm",
                hx_post="/save",
                hx_include="this",
                hx_target="#save-status",
                hx_swap="innerHTML",
            ),
            Div(id="save-status"),
            cls="note",
        ),
        Div(
            H3("Model output"), right_top_component, cls="output"
        ),  # output now last
        cls="right",
    )

    return (*head_content, Body(Div(left_panel, right_panel, cls="grid")))


@rt("/save")
async def save_note(request: Request):
    """Explicit save endpoint called by HTMX or Save button."""
    form = await request.form()
    idx = int(form["idx"])
    note = textwrap.dedent(str(form["note"]))
    store.set(idx, note)
    store.save()
    return P("Saved!", id="save-status")


@rt("/navigate")
async def navigate(request: Request):
    """
    Save current note then redirect to next/previous item.
    HTMX automatically swaps the <body> with the redirect response.
    """
    form = await request.form()
    try:
        # Fallback to 0 if 'idx' is missing or not a valid integer (e.g., empty string).
        idx = int(form.get("idx", "0"))
    except ValueError:
        idx = 0

    direction = form.get("direction", "next")
    # Use .get() for note to avoid errors if it's missing from the form.
    note = textwrap.dedent(str(form.get("note", "")))
    store.set(idx, note)
    store.save()

    if direction == "prev":
        idx = max(0, idx - 1)
    else:
        idx = min(len(SOURCE_DATA) - 1, idx + 1)

    return RedirectResponse(f"/item/{idx}")


if __name__ == "__main__":
    serve(port=5001, reload=True)
