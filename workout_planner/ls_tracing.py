"""LangSmith tracing setup for the workout planner (no Streamlit dependency).

What this module fixes (all of these used to fail silently):

* keys that only live in ``st.secrets`` / ``.env`` and never reach ``os.environ``
* legacy ``LANGCHAIN_*`` names vs. new ``LANGSMITH_*`` names
* quoted / padded keys and placeholder keys copied from ``example.env``
* EU-region keys sent to the US endpoint (401/403 on every upload)
* huge base64 photos / audio clips copied into every run, which makes the
  ingestion payload exceed LangSmith's limits so the whole trace is dropped

Call :func:`bootstrap` once, BEFORE importing anything from ``langchain``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from itertools import islice
from typing import Any, Iterator, Mapping

logger = logging.getLogger("workout_planner.tracing")

US_ENDPOINT = "https://api.smith.langchain.com"
EU_ENDPOINT = "https://eu.api.smith.langchain.com"

_PLACEHOLDER_MARKERS = ("your_", "****", "_here")
_TRUTHY = {"1", "true", "yes", "on"}

# Any single string longer than this is cut before it is uploaded to LangSmith.
MAX_STR_CHARS = 20_000


@dataclass(frozen=True)
class TracingStatus:
    enabled: bool
    project: str | None = None
    endpoint: str | None = None
    note: str = ""  # why tracing is off, or which region was auto-detected


_STATUS: TracingStatus | None = None


# --------------------------------------------------------------------------- #
# Payload shrinking (hide_inputs / hide_outputs)
# --------------------------------------------------------------------------- #
def shrink(obj: Any) -> Any:
    """Return a copy of ``obj`` where oversized strings (base64 media) are cut.

    Works on plain JSON-like data and on LangChain messages (objects with a
    ``content`` attribute), because chain inputs still contain message objects
    when they reach LangSmith's ``hide_inputs`` hook.
    """
    if isinstance(obj, str):
        if len(obj) <= MAX_STR_CHARS:
            return obj
        return f"{obj[:200]}… [truncated for tracing, {len(obj):,} chars]"
    if isinstance(obj, (bytes, bytearray)):
        return f"[{len(obj):,} bytes omitted from trace]"
    if isinstance(obj, Mapping):
        return {k: shrink(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [shrink(v) for v in obj]
    if hasattr(obj, "content") and hasattr(obj, "model_copy"):  # LangChain message
        try:
            return obj.model_copy(update={"content": shrink(obj.content)})
        except Exception:  # pragma: no cover - never break the app for tracing
            return obj
    return obj


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #
def _flatten(secrets: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    for key, value in secrets.items():
        if isinstance(value, Mapping):  # [section] tables in secrets.toml
            yield from _flatten(value)
        elif isinstance(value, (str, int, float, bool)):
            yield key, value


def _is_placeholder(value: str) -> bool:
    low = value.lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def _probe(endpoint: str, key: str) -> int | None:
    """HTTP status of an authenticated call, or None if the network is down."""
    try:
        import httpx

        headers = {"x-api-key": key}
        if os.environ.get("LANGSMITH_WORKSPACE_ID"):
            headers["x-tenant-id"] = os.environ["LANGSMITH_WORKSPACE_ID"]
        return httpx.get(f"{endpoint}/sessions", params={"limit": 1}, headers=headers, timeout=4).status_code
    except Exception:
        return None


def _detect_region(key: str) -> str:
    """If the key is rejected by the US endpoint but accepted by the EU one, use EU."""
    if _probe(US_ENDPOINT, key) in (401, 403) and _probe(EU_ENDPOINT, key) == 200:
        os.environ["LANGSMITH_ENDPOINT"] = EU_ENDPOINT
        return "EU region detected automatically"
    return ""


def bootstrap(secrets: Mapping[str, Any] | None = None, *, force: bool = False) -> TracingStatus:
    """Normalise the environment and switch tracing on/off consistently.

    Safe to call on every Streamlit rerun: the work is done once per process.
    """
    global _STATUS
    if _STATUS is not None and not force:
        return _STATUS

    # 1. Deployment secrets -> environment (never override a real env var).
    for key, value in _flatten(secrets or {}):
        if not os.environ.get(key):
            os.environ[key] = str(value)

    # 2. Strip quotes / whitespace that make auth fail silently.
    for key in list(os.environ):
        if key.startswith(("LANGSMITH_", "LANGCHAIN_")) or key.endswith("_API_KEY"):
            os.environ[key] = os.environ[key].strip().strip("'\"").strip()

    # 3. Legacy names -> new names.
    for new, old in (
        ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
        ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
        ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
        ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"),
    ):
        if not os.environ.get(new) and os.environ.get(old):
            os.environ[new] = os.environ[old]

    # 4. Decide whether tracing is on.
    key = os.environ.get("LANGSMITH_API_KEY", "")
    note = ""
    if key and _is_placeholder(key):
        note = "LANGSMITH_API_KEY still has the placeholder value from the example file"
        key = ""
    explicit = os.environ.get("LANGSMITH_TRACING", "").strip().lower()
    if explicit:
        wanted = explicit in _TRUTHY
        if not wanted:
            note = note or "Tracing disabled (LANGSMITH_TRACING=false)"
    else:
        wanted = bool(key)
    if wanted and not key:
        note = note or "Add LANGSMITH_API_KEY to your app Secrets or .env"
        wanted = False

    os.environ.setdefault("LANGSMITH_PROJECT", "workout-planner")
    project = os.environ["LANGSMITH_PROJECT"]

    if not wanted:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        _STATUS = TracingStatus(False, project, None, note)
        return _STATUS

    # 5. Tracing is on: set both name families so every installed version agrees.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = project

    if not os.environ.get("LANGSMITH_ENDPOINT"):
        note = _detect_region(key)
    if os.environ.get("LANGSMITH_ENDPOINT"):
        os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]

    # 6. One shared client for LangChain's tracer AND @traceable, with media
    #    stripped from payloads. Must exist before the first run is created.
    try:
        from langsmith import run_trees

        run_trees.get_cached_client(hide_inputs=shrink, hide_outputs=shrink)
    except Exception:
        logger.warning("Could not install the LangSmith payload filter", exc_info=True)

    _STATUS = TracingStatus(True, project, os.environ.get("LANGSMITH_ENDPOINT") or US_ENDPOINT, note)
    return _STATUS


def status() -> TracingStatus:
    return _STATUS or bootstrap()


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #
def verify() -> tuple[str, str]:
    """Return (state, detail) with state in ok | off | error, using a real auth call."""
    st = status()
    if not st.enabled:
        return "off", st.note or "Tracing is disabled"
    try:
        from langsmith import run_trees

        list(islice(run_trees.get_cached_client().list_projects(limit=1), 1))
    except Exception as exc:
        text = str(exc)
        if "401" in text or "nauthorized" in text:
            hint = "Invalid or revoked LANGSMITH_API_KEY"
        elif "403" in text or "orbidden" in text:
            hint = "Forbidden: wrong region (set LANGSMITH_ENDPOINT) or workspace (set LANGSMITH_WORKSPACE_ID)"
        else:
            hint = text
        return "error", f"{hint} [{text[:100]}]" if hint != text else text[:160]
    detail = st.project or ""
    return "ok", f"{detail} · {st.note}" if st.note else detail


def flush() -> None:
    """Upload pending traces now (Streamlit may stop the process at any time)."""
    try:
        from langsmith import run_trees

        if run_trees._CLIENT is not None:  # noqa: SLF001
            run_trees._CLIENT.flush()  # noqa: SLF001
    except Exception:
        logger.warning("LangSmith flush failed", exc_info=True)
