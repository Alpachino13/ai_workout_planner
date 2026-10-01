"""A Streamlit-based UI for the Workout Planner Supervisor Agent."""

import asyncio
import base64
import os
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Dict

import streamlit as st
from dotenv import load_dotenv

# 1. Suppress harmless LangChain schema translation warnings
warnings.filterwarnings("ignore", message=".*is not supported in schema.*")

# Load .env FIRST
load_dotenv()

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
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.runtime import Runtime
from langgraph.types import Command
from tavily import TavilyClient

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
        return "*(No profile data saved yet)*"
    return "\n".join(f"- **{k.replace('_', ' ').title()}**: {v}" for k, v in profile.items())

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
    stripped = [_strip_media(m) for i, m in enumerate(messages) if i < last_human and m.type == "human" and _has_media(m)]
    return {"messages": stripped} if stripped else None

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

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

research_subagent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt=(
        "You are a fitness research specialist. "
        "Use web search to find accurate, up-to-date fitness information. "
        "Summarize your findings clearly and concisely."
    )
)

@tool("research_assistant")
def call_research_assistant(query: str) -> str:
    """Delegates research queries to a subagent to find fitness, exercise, or equipment information."""
    result = research_subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

MCP_SERVERS = {
    "rhylthyme": {
        "transport": "streamable_http",
        "url": "https://mcp.rhylthyme.com/mcp",
    },
}

async def build_agent():
    tools = [call_research_assistant, update_profile]
    try:
        client = MultiServerMCPClient(MCP_SERVERS)
        mcp_tools = await client.get_tools()
        tools.extend(mcp_tools)
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
        "run_name": "workout_planner_turn",
        "tags": ["workout-planner", "gemini"],
    }

# ==============================================================================
# STREAMLIT UI SETUP & ASYNC HELPERS
# ==============================================================================

st.set_page_config(page_title="Workout Planner AI", page_icon="💪", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    .stChatInputContainer { padding-bottom: 20px; }
    div[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e9ecef; }
    .stButton>button { border-radius: 6px; font-weight: 500; }
    .hitl-box { background-color: #e3f2fd; padding: 15px; border-radius: 8px; border: 1px solid #90caf9; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

def run_async(coro):
    """Safely run async functions inside Streamlit."""
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
    """Extracts raw text from a list of blocks or returns the string directly."""
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict) and "text" in block)
    return str(content)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hitl_snapshot" not in st.session_state:
    st.session_state.hitl_snapshot = None

with st.spinner("Initializing AI Agent & Connecting to Tools..."):
    agent = get_cached_agent()

thread_id = st.session_state.thread_id
config = make_config(thread_id)

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.title("💪 Workout Planner")
    
    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.hitl_snapshot = None
        st.rerun()

    st.divider()
    st.subheader("⚙️ Settings")
    selected_units = st.selectbox("Units", ["metric", "imperial"], index=0)
    selected_lang = st.text_input("Language", value="English")
    ctx = Context(units=selected_units, language=selected_lang)

    st.divider()
    st.subheader("👤 User Profile")
    current_state = run_async(agent.aget_state(config))
    profile_data = current_state.values.get("profile") if current_state and hasattr(current_state, 'values') else {}
    st.markdown(format_profile(profile_data))
    
    st.divider()
    st.subheader("📎 Attach Media")
    uploaded_file = st.file_uploader("Upload Image or Audio", type=["jpg", "png", "jpeg", "mp3", "wav"])

# ==============================================================================
# MAIN CHAT INTERFACE
# ==============================================================================

st.title("AI Fitness Supervisor")
st.markdown("I coordinate your workout plans using subagents and tools. Let's get started!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- HUMAN IN THE LOOP (HITL) UI ---
if st.session_state.hitl_snapshot and st.session_state.hitl_snapshot.next:
    last_msg = st.session_state.hitl_snapshot.values["messages"][-1]
    
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        st.markdown('<div class="hitl-box">', unsafe_allow_html=True)
        st.subheader("🛡️ Agent Action Requires Approval")
        st.write("The supervisor wants to execute the following tools:")
        
        for tc in last_msg.tool_calls:
            st.code(f"Tool: {tc['name']}\nArguments: {tc['args']}", language="json")
        
        col1, col2, col3 = st.columns(3)
        
        if col1.button("✅ Approve", use_container_width=True, type="primary"):
            with st.spinner("Executing tools..."):
                run_async(agent.ainvoke(None, config, context=ctx))
                new_state = run_async(agent.aget_state(config))
                if not new_state.next:
                    # Applied extract_text here
                    clean_text = extract_text(new_state.values["messages"][-1].content)
                    st.session_state.messages.append({"role": "assistant", "content": clean_text})
                st.session_state.hitl_snapshot = run_async(agent.aget_state(config))
            st.rerun()

        if col2.button("🚫 Reject", use_container_width=True):
            with st.spinner("Rejecting..."):
                reject_msgs = [
                    ToolMessage(tool_call_id=tc["id"], name=tc["name"], content="Action blocked: User rejected.") 
                    for tc in last_msg.tool_calls
                ]
                run_async(agent.aupdate_state(config, {"messages": reject_msgs}, as_node="tools"))
                run_async(agent.ainvoke(None, config, context=ctx))
                new_state = run_async(agent.aget_state(config))
                if not new_state.next:
                    # Applied extract_text here
                    clean_text = extract_text(new_state.values["messages"][-1].content)
                    st.session_state.messages.append({"role": "assistant", "content": clean_text})
                st.session_state.hitl_snapshot = run_async(agent.aget_state(config))
            st.rerun()

        feedback = col3.popover("✍️ Edit").text_input("What should it do differently?")
        if feedback:
            with st.spinner("Updating AI with feedback..."):
                feedback_msgs = [
                    ToolMessage(tool_call_id=tc["id"], name=tc["name"], content=f"Action blocked. User feedback: {feedback}") 
                    for tc in last_msg.tool_calls
                ]
                run_async(agent.aupdate_state(config, {"messages": feedback_msgs}, as_node="tools"))
                run_async(agent.ainvoke(None, config, context=ctx))
                new_state = run_async(agent.aget_state(config))
                if not new_state.next:
                    # Applied extract_text here
                    clean_text = extract_text(new_state.values["messages"][-1].content)
                    st.session_state.messages.append({"role": "assistant", "content": clean_text})
                st.session_state.hitl_snapshot = run_async(agent.aget_state(config))
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()


# --- CHAT INPUT ---
if user_text := st.chat_input("Describe your goals, or ask for a workout plan..."):
    
    content = []
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        b64_data = base64.b64encode(file_bytes).decode("utf-8")
        mime = uploaded_file.type
        kind = "image" if mime.startswith("image") else "audio"
        content.append({"type": kind, "base64": b64_data, "mime_type": mime})
        st.session_state.messages.append({"role": "user", "content": f"*[Uploaded {kind}: {uploaded_file.name}]*\n{user_text}"})
    else:
        st.session_state.messages.append({"role": "user", "content": user_text})
    
    content.append({"type": "text", "text": user_text})

    with st.chat_message("user"):
        st.markdown(st.session_state.messages[-1]["content"])

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                run_async(agent.ainvoke({"messages": [{"role": "user", "content": content}]}, config, context=ctx))
                
                snapshot = run_async(agent.aget_state(config))
                st.session_state.hitl_snapshot = snapshot
                
                if snapshot.next:
                    st.rerun()
                else:
                    # Applied extract_text here
                    final_msg = extract_text(snapshot.values["messages"][-1].content)
                    st.session_state.messages.append({"role": "assistant", "content": final_msg})
                    st.markdown(final_msg)
            
            except Exception as e:
                st.error(f"Error: {e}")