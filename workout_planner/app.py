"""Workout Planner — Streamlit UI for the supervisor agent (LangChain + Gemini + LangSmith)."""

import asyncio
import base64
import html
import os
import uuid
import warnings
from dataclasses import dataclass
from itertools import islice
from typing import Any, Dict

import streamlit as st

# Must be the very first Streamlit call.
st.set_page_config(page_title="Workout Planner", page_icon="💪", layout="wide")

from dotenv import load_dotenv

warnings.filterwarnings("ignore", message=".*is not supported in schema.*")
load_dotenv()

# ==============================================================================
# ENVIRONMENT + LANGSMITH TRACING
# ------------------------------------------------------------------------------
# Why tracing was silent: on Streamlit Cloud your local .env is NOT deployed.
# Keys live in  App settings -> Secrets.  We copy them into os.environ here,
# normalise the old LANGCHAIN_* / new LANGSMITH_* names, and force tracing on
# BEFORE any LangChain object is created.
# ==============================================================================


def bootstrap_env() -> None:
    try:
        for key, value in dict(st.secrets).items():
            if isinstance(value, (str, int, float, bool)) and not os.environ.get(key):
                os.environ[key] = str(value)
    except Exception:
        pass  # no secrets.toml locally — fine

    # Strip stray quotes/whitespace that break auth silently.
    for key in list(os.environ):
        if key.startswith(("LANGSMITH_", "LANGCHAIN_")):
            os.environ[key] = os.environ[key].strip().strip("'\"")

    aliases = [
        ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
        ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT"),
        ("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT"),
    ]
    for new, old in aliases:
        if not os.environ.get(new) and os.environ.get(old):
            os.environ[new] = os.environ[old]

    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "workout-planner")

    # Set both names so any installed langsmith/langchain version picks it up.
    if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGSMITH_API_KEY", "")
        os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
        if os.environ.get("LANGSMITH_ENDPOINT"):
            os.environ["LANGCHAIN_ENDPOINT"] = os.environ["LANGSMITH_ENDPOINT"]


bootstrap_env()

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    ModelRequest,
    SummarizationMiddleware,
    before_model,
    dynamic_prompt,
)
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command
from tavily import TavilyClient


@st.cache_data(ttl=300, show_spinner=False)
def check_langsmith() -> tuple[str, str]:
    """Returns (status, detail): ok | off | error."""
    if os.environ.get("LANGSMITH_TRACING", "").lower() != "true":
        return "off", "Add LANGSMITH_API_KEY to your app Secrets"
    try:
        from langsmith import Client

        list(islice(Client().list_projects(limit=1), 1))  # real auth round-trip
        return "ok", os.environ["LANGSMITH_PROJECT"]
    except Exception as e:  # bad key, wrong region endpoint, network...
        return "error", str(e)[:160]


def flush_traces() -> None:
    """Make sure traces are uploaded before the script returns."""
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception:
        pass


# ==============================================================================
# CORE AGENT LOGIC
# ==============================================================================

tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for up-to-date fitness, exercise, or equipment information."""
    return tavily_client.search(query, max_results=5)


@dataclass
class Context:
    units: str = "metric"
    language: str = "English"


class WorkoutState(AgentState):
    profile: dict[str, Any]


@tool
def update_profile(
    runtime: ToolRuntime[Context, WorkoutState],
    goal: str | None = None,
    experience_level: str | None = None,
    days_per_week: int | None = None,
    session_minutes: int | None = None,
    equipment: str | None = None,
    limitations: str | None = None,
    physique_notes: str | None = None,
) -> Command:
    """Save or update facts about the user in short phrases. Only pass fields that are new or changed."""
    updates = {
        key: value
        for key, value in {
            "goal": goal,
            "experience_level": experience_level,
            "days_per_week": days_per_week,
            "session_minutes": session_minutes,
            "equipment": equipment,
            "limitations": limitations,
            "physique_notes": physique_notes,
        }.items()
        if value is not None
    }
    profile = {**(runtime.state.get("profile") or {}), **updates}
    return Command(
        update={
            "profile": profile,
            "messages": [
                ToolMessage(
                    content=f"Profile updated: {', '.join(updates) or 'nothing changed'}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


def format_profile(profile: dict[str, Any]) -> str:
    if not profile:
        return "No profile data saved yet."
    return "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in profile.items())


SYSTEM_PROMPT = """
You are a careful, encouraging workout-planning supervisor.

Create practical workout plans based on the user's goals, experience, schedule,
available equipment, time, preferences, and limitations. You coordinate this
process by delegating specialized tasks to your subagents. Ask focused follow-up
questions when important information is missing.

Safety rules:
- Do not diagnose injuries or medical conditions.
- Encourage the user to consult a qualified clinician for major exercise restrictions.

Images and audio:
- Use media only to understand the starting point and target.
- Never estimate body-fat percentage or weight precisely.

Tools & Delegation:
- Delegate queries to your `research_assistant` subagent to verify exercise guidance.
- Save essentials with update_profile as you learn them.
- You may have Rhylthyme scheduling tools. Use them only after user approval.

When the user asks for a plan, present it in a clean day-by-day format.
"""


@dynamic_prompt
async def build_system_prompt(request: ModelRequest) -> str:
    ctx = request.runtime.context or Context()
    profile = request.state.get("profile") or {}
    parts = [SYSTEM_PROMPT, f"Session settings: use {ctx.units} units; reply in {ctx.language}."]
    if profile:
        parts.append("Saved user profile:\n" + format_profile(profile))
    else:
        parts.append("No user profile saved yet. Gather the essentials, then save them.")
    return "\n\n".join(parts)


MEDIA_TYPES = {"image", "audio"}


def _has_media(message: Any) -> bool:
    return isinstance(message.content, list) and any(
        isinstance(b, dict) and b.get("type") in MEDIA_TYPES for b in message.content
    )


def _strip_media(message: Any) -> Any:
    blocks = [
        {"type": "text", "text": f"[{b['type']} shared earlier; key details are in the saved profile]"}
        if isinstance(b, dict) and b.get("type") in MEDIA_TYPES
        else b
        for b in message.content
    ]
    return message.model_copy(update={"content": blocks})


@before_model
def drop_old_media(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    messages = state["messages"]
    last_human = max((i for i, m in enumerate(messages) if m.type == "human"), default=-1)
    stripped = [
        _strip_media(m)
        for i, m in enumerate(messages)
        if i < last_human and m.type == "human" and _has_media(m)
    ]
    return {"messages": stripped} if stripped else None


model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

research_subagent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=(
        "You are a fitness research specialist. "
        "Use web search to find accurate, up-to-date fitness information. "
        "Summarize your findings clearly and concisely."
    ),
)


@tool("research_assistant")
def call_research_assistant(query: str) -> str:
    """Delegates research queries to a subagent to find fitness, exercise, or equipment information."""
    result = research_subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return extract_text(result["messages"][-1].content)


MCP_SERVERS = {
    "rhylthyme": {
        "transport": "streamable_http",
        "url": "https://mcp.rhylthyme.com/mcp",
    },
}

# Tools that run without asking. Everything else (e.g. Rhylthyme scheduling) needs approval.
SAFE_TOOLS = {"update_profile", "research_assistant"}


async def build_agent():
    tools = [call_research_assistant, update_profile]
    try:
        client = MultiServerMCPClient(MCP_SERVERS)
        tools.extend(await client.get_tools())
    except Exception:
        pass

    return create_agent(
        model=model,
        tools=tools,
        state_schema=WorkoutState,
        context_schema=Context,
        middleware=[
            drop_old_media,
            SummarizationMiddleware(model=model, trigger=("tokens", 6000), keep=("messages", 12)),
            build_system_prompt,
        ],
        checkpointer=InMemorySaver(),
        interrupt_before=["tools"],
    )


def make_config(thread_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},  # groups turns into one LangSmith thread
        "run_name": "workout_planner_turn",
        "tags": ["workout-planner", "gemini", "streamlit"],
    }


# ==============================================================================
# DESIGN SYSTEM
# ==============================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --ink: #0E1726;
  --ink-soft: #52607A;
  --chalk: #F4F6FA;
  --paper: #FFFFFF;
  --line: #E3E8F0;
  --cobalt: #2B50F0;
  --cobalt-soft: #EAEFFE;
  --good: #12805C;
  --bad: #C2362B;
}

html, body, [class*="css"], .stApp { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }
.stApp { background: var(--chalk); color: var(--ink); }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 880px; padding-top: 2.2rem; padding-bottom: 7rem; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--paper); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }
.brand { display:flex; align-items:center; gap:.7rem; margin-bottom:1.1rem; }
.brand-mark { width:38px; height:38px; border-radius:11px; background:var(--cobalt);
  display:grid; place-items:center; font-size:1.2rem; }
.brand-name { font-weight:800; font-size:1.08rem; letter-spacing:-.01em; line-height:1.1; }
.brand-sub { font-size:.74rem; color:var(--ink-soft); font-weight:500; }
.side-title { font-size:.8rem; font-weight:700; color:var(--ink-soft); margin:1.3rem 0 .5rem; }

/* Profile */
.profile { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:var(--paper); }
.profile-row { padding:.6rem .85rem; border-bottom:1px solid var(--line); }
.profile-row:last-child { border-bottom:none; }
.profile-k { font-size:.72rem; color:var(--ink-soft); font-weight:600; }
.profile-v { font-size:.88rem; font-weight:500; color:var(--ink); margin-top:.1rem; }
.profile-empty { border:1px dashed var(--line); border-radius:12px; padding:.9rem; font-size:.85rem; color:var(--ink-soft); }

/* Status */
.status { display:flex; align-items:center; gap:.5rem; font-size:.8rem; font-weight:600; }
.dot { width:8px; height:8px; border-radius:50%; background:#9AA6B8; flex:none; }
.dot.ok { background:var(--good); } .dot.error { background:var(--bad); }
.status-detail { font-size:.74rem; color:var(--ink-soft); margin:.2rem 0 0 1.05rem; word-break:break-word; }

/* Hero / empty state */
.hero h1 { font-size:2.5rem; font-weight:800; letter-spacing:-.03em; line-height:1.1; margin:0 0 .6rem; color:var(--ink); }
.hero p { font-size:1.05rem; color:var(--ink-soft); max-width:34rem; line-height:1.55; margin:0 0 1.8rem; }

/* Chat */
[data-testid="stChatMessage"] { background:var(--paper); border:1px solid var(--line); border-radius:16px;
  padding:1rem 1.2rem; margin-bottom:.8rem; box-shadow:0 1px 2px rgba(14,23,38,.04); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:var(--cobalt-soft); border-color:#D3DCFB; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color:var(--ink); line-height:1.65; font-size:.97rem; }
[data-testid="stChatMessage"] h1, [data-testid="stChatMessage"] h2, [data-testid="stChatMessage"] h3 { letter-spacing:-.01em; margin-top:1rem; }

/* Input */
[data-testid="stChatInput"] { border:1px solid var(--line); border-radius:16px; background:var(--paper);
  box-shadow:0 8px 28px rgba(14,23,38,.08); }
[data-testid="stChatInput"]:focus-within { border-color:var(--cobalt); }
[data-testid="stChatInput"] textarea { color:var(--ink); }

/* Buttons */
.stButton > button { border-radius:10px; font-weight:600; border:1px solid var(--line); background:var(--paper);
  color:var(--ink); transition:border-color .15s, background .15s; }
.stButton > button:hover { border-color:var(--cobalt); color:var(--cobalt); background:var(--paper); }
.stButton > button[kind="primary"] { background:var(--cobalt); border-color:var(--cobalt); color:#fff; }
.stButton > button[kind="primary"]:hover { background:#1F3FD0; color:#fff; }
.stButton > button:focus-visible { outline:2px solid var(--cobalt); outline-offset:2px; }

/* Approval card */
.approval-title { font-weight:700; font-size:1.02rem; margin-bottom:.15rem; }
.approval-sub { color:var(--ink-soft); font-size:.88rem; margin-bottom:.7rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px; }

@media (prefers-reduced-motion: reduce) { * { transition:none !important; animation:none !important; } }
@media (max-width: 640px) { .hero h1 { font-size:1.8rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

STARTERS = [
    ("Build muscle at home", "I want to build muscle with just dumbbells at home. Can you plan my week?"),
    ("Lose fat, 3 days a week", "I want to lose fat and can train 3 days a week, 45 minutes per session."),
    ("First time in the gym", "I'm a beginner starting at the gym. Where should I begin?"),
    ("Improve my endurance", "I want to improve my running endurance. What should my plan look like?"),
]

# ==============================================================================
# ASYNC + STATE HELPERS
# ==============================================================================


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@st.cache_resource(show_spinner=False)
def get_cached_agent():
    return run_async(build_agent())


def extract_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and "text" in b)
    return str(content)


def init_state() -> None:
    st.session_state.setdefault("thread_id", str(uuid.uuid4()))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("hitl_snapshot", None)
    st.session_state.setdefault("uploader_key", 0)


def reset_conversation() -> None:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.hitl_snapshot = None
    st.session_state.uploader_key += 1


init_state()
with st.spinner("Starting your coach..."):
    agent = get_cached_agent()
config = make_config(st.session_state.thread_id)


def advance(ctx: Context):
    """Keep running while pending tools are safe; stop when approval is needed or the turn ends."""
    state = run_async(agent.aget_state(config))
    for _ in range(8):
        if not state.next:
            break
        calls = getattr(state.values["messages"][-1], "tool_calls", None) or []
        if calls and not all(c["name"] in SAFE_TOOLS for c in calls):
            break
        run_async(agent.ainvoke(None, config, context=ctx))
        state = run_async(agent.aget_state(config))
    flush_traces()
    st.session_state.hitl_snapshot = state
    if not state.next:
        st.session_state.messages.append(
            {"role": "assistant", "content": extract_text(state.values["messages"][-1].content)}
        )
    return state


def resolve_pending(ctx: Context, tool_calls, reason: str) -> None:
    msgs = [ToolMessage(tool_call_id=tc["id"], name=tc["name"], content=reason) for tc in tool_calls]
    run_async(agent.aupdate_state(config, {"messages": msgs}, as_node="tools"))
    run_async(agent.ainvoke(None, config, context=ctx))
    advance(ctx)


# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">💪</div><div>'
        '<div class="brand-name">Workout Planner</div>'
        '<div class="brand-sub">Your AI training coach</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("New conversation", use_container_width=True, type="primary", icon=":material/add:"):
        reset_conversation()
        st.rerun()

    st.markdown('<div class="side-title">Preferences</div>', unsafe_allow_html=True)
    units = st.selectbox("Units", ["metric", "imperial"], index=0)
    language = st.text_input("Reply language", value="English")
    ctx = Context(units=units, language=language)

    st.markdown('<div class="side-title">Your profile</div>', unsafe_allow_html=True)
    state_now = run_async(agent.aget_state(config))
    profile = (state_now.values.get("profile") if state_now and hasattr(state_now, "values") else None) or {}
    if profile:
        rows = "".join(
            f'<div class="profile-row"><div class="profile-k">{html.escape(k.replace("_", " ").title())}</div>'
            f'<div class="profile-v">{html.escape(str(v))}</div></div>'
            for k, v in profile.items()
        )
        st.markdown(f'<div class="profile">{rows}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="profile-empty">Your goals and equipment will appear here as you chat.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="side-title">Photo or voice note</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Attach",
        type=["jpg", "png", "jpeg", "mp3", "wav"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
        help="Share your current physique or goal. It's used with your next message only.",
    )

    status, detail = check_langsmith()
    label = {"ok": "Tracing on", "off": "Tracing off", "error": "Tracing error"}[status]
    st.markdown('<div class="side-title">Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="status"><span class="dot {status}"></span>{label}</div>'
        f'<div class="status-detail">{html.escape(detail)}</div>',
        unsafe_allow_html=True,
    )

# ==============================================================================
# MAIN
# ==============================================================================

pending = st.session_state.hitl_snapshot
awaiting_approval = bool(pending and pending.next)

if not st.session_state.messages and not awaiting_approval:
    st.markdown(
        '<div class="hero"><h1>Train with a plan that fits your week.</h1>'
        "<p>Tell me your goal, your equipment and how much time you have. "
        "I'll build a day-by-day plan and check it against current guidance.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (title, prompt_text) in enumerate(STARTERS):
        if cols[i % 2].button(title, key=f"starter_{i}", use_container_width=True):
            st.session_state.queued_prompt = prompt_text
            st.rerun()
else:
    for msg in st.session_state.messages:
        avatar = ":material/person:" if msg["role"] == "user" else ":material/fitness_center:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# --- Approval card ------------------------------------------------------------
if awaiting_approval:
    last_msg = pending.values["messages"][-1]
    calls = getattr(last_msg, "tool_calls", None) or []
    if calls:
        with st.container(border=True):
            st.markdown(
                '<div class="approval-title">Approval needed</div>'
                '<div class="approval-sub">The coach wants to run the actions below. Nothing happens until you approve.</div>',
                unsafe_allow_html=True,
            )
            for tc in calls:
                with st.expander(tc["name"].replace("_", " ").title(), expanded=True):
                    st.json(tc["args"])

            c1, c2, c3 = st.columns(3)
            if c1.button("Approve", type="primary", use_container_width=True, icon=":material/check:"):
                with st.spinner("Running..."):
                    run_async(agent.ainvoke(None, config, context=ctx))
                    advance(ctx)
                st.rerun()
            if c2.button("Decline", use_container_width=True, icon=":material/close:"):
                with st.spinner("Updating..."):
                    resolve_pending(ctx, calls, "Action blocked: user declined.")
                st.rerun()
            with c3.popover("Suggest a change", use_container_width=True, icon=":material/edit:"):
                feedback = st.text_input("What should change?", key="hitl_feedback")
                if st.button("Send", key="hitl_send", disabled=not feedback):
                    with st.spinner("Updating..."):
                        resolve_pending(ctx, calls, f"Action blocked. User feedback: {feedback}")
                    st.rerun()
        st.stop()

# --- Chat input -----------------------------------------------------------------
user_text = st.chat_input("Describe your goals or ask for a plan") or st.session_state.pop("queued_prompt", None)

if user_text:
    content = []
    shown = user_text
    if uploaded_file is not None:
        b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
        mime = uploaded_file.type
        kind = "image" if mime.startswith("image") else "audio"
        content.append({"type": kind, "base64": b64, "mime_type": mime})
        shown = f"*Attached {kind}: {uploaded_file.name}*\n\n{user_text}"
    content.append({"type": "text", "text": user_text})
    st.session_state.messages.append({"role": "user", "content": shown})

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(shown)
    with st.chat_message("assistant", avatar=":material/fitness_center:"):
        with st.spinner("Thinking..."):
            try:
                run_async(agent.ainvoke({"messages": [{"role": "user", "content": content}]}, config, context=ctx))
                advance(ctx)
                st.session_state.uploader_key += 1  # clear the attachment after it's sent
                st.rerun()
            except Exception as e:
                flush_traces()
                st.error(f"Something went wrong: {e}. Try sending your message again.")
