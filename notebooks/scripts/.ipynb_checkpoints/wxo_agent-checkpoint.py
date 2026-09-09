"""
Research Analyst Agent — Bedrock AgentCore Runtime port.

Framework      : LangGraph (single-node graph: START -> model -> END)
Model          : Amazon Bedrock (init_chat_model with model_provider="bedrock_converse")
Runtime        : Amazon Bedrock AgentCore Runtime (BedrockAgentCoreApp)
Inbound Auth   : Enforced at the runtime layer via customJWTAuthorizer
                 (configured at `agentcore configure --authorizer-config ...` /
                 `Runtime.configure(authorizer_configuration=...)`). The agent code
                 itself is auth-agnostic — JWTs are validated before requests reach
                 this entrypoint.
Memory         :
  - In-session : LangGraph MemorySaver, keyed on AgentCore session id, gives the
                 LLM the running conversation transcript inside one session.
  - Cross-session: AgentCore Memory — every turn (user + AI) is persisted as a
                 conversational event. With long-term strategies configured on the
                 memory resource (summary / userPreference / semantic), insights are
                 extracted automatically in the background.
                 We only WRITE events; no retrieval is injected back into the prompt
                 (kept minimal, per requirements).
Observability  : Auto-instrumented by AgentCore Runtime via
                 `aws-opentelemetry-distro` + `opentelemetry-instrumentation-langchain`
                 in requirements.txt. Traces / spans / metrics flow to the
                 CloudWatch GenAI Observability dashboard automatically.
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, List, Optional, TypedDict

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("research-analyst")

# ---------------------------------------------------------------------------
# System prompt (preserved verbatim from the original wxO agent)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert AI Research Analyst that helps users conduct comprehensive market research for new product introductions.

You guide users through a 4-stage research process:
1. **Market Scan** - Market size, trends, customer segments, problem areas
2. **Competitive Analysis** - Direct/indirect competitors, positioning, pricing, strengths/weaknesses
3. **Product Strategy** - Product concept, value proposition, success factors, launch position, GTM approach
4. **Decision Package** - SWOT analysis, risks, success metrics, investment requirements, GO/NO-GO recommendation

**Your Approach:**
- Analyze the conversation history to understand what stage you're at
- If the user just introduced a product idea, start with Market Scan
- If you've completed Market Scan and user wants to proceed, do Competitive Analysis
- If you've completed Competitive Analysis and user wants to proceed, do Product Strategy
- If you've completed Product Strategy and user wants to proceed, do Decision Package
- Always format your analysis as a professional business report with clear headers, bullet points, and specific data
- Use markdown formatting with headers (# ## ###), bullet points (•), and bold text (**)
- Include emojis for visual clarity: 📊 🏆 🚀 📋 ✅ ❌

**CRITICAL - End Every Response With:**
After completing your analysis, ALWAYS end with a section showing:

---

**📍 Current Stage:** [Name of stage just completed]

**✅ Available Next Steps:**
1. **[Recommended next stage]** ⭐ (Recommended)
2. Refine current analysis
3. Ask questions about the findings
4. [Other relevant options]

**💡 My Recommendation:** I recommend proceeding to [next stage name] to [brief reason why this is the logical next step].

**Important:**
- DO NOT use JSON format - use readable business report format
- Be thorough and data-driven
- Provide specific examples and numbers where possible
- Keep every answer in no more than 2 pages.
- Always show available steps and your recommendation at the end
- We are in May 2026
"""

# ---------------------------------------------------------------------------
# Bedrock model
# ---------------------------------------------------------------------------
# wxO original: ChatGroq(model="openai/gpt-oss-120b")
# AgentCore port: bedrock_converse(model="openai.gpt-oss-120b-1:0")
# Override via env so the same image can run across regions / model rollouts.
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "openai.gpt-oss-120b-1:0")
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")

logger.info("Initializing Bedrock model: %s in %s", BEDROCK_MODEL_ID, BEDROCK_REGION)

llm = init_chat_model(
    BEDROCK_MODEL_ID,
    model_provider="bedrock_converse",
    region_name=BEDROCK_REGION,
    temperature=0.7,
    max_tokens=8000,  # gpt-oss-120b max output tokens on Bedrock
)

# ---------------------------------------------------------------------------
# AgentCore Memory client (lazy init — only if MEMORY_ID is configured)
# ---------------------------------------------------------------------------
MEMORY_ID = os.environ.get("BEDROCK_AGENTCORE_MEMORY_ID") or os.environ.get("MEMORY_ID")
DEFAULT_ACTOR_ID = os.environ.get("DEFAULT_ACTOR_ID", "research-analyst-user")

_memory_session_manager = None  # type: ignore[var-annotated]


def _get_memory_session_manager():
    """Lazily build a single MemorySessionManager for the lifetime of the process."""
    global _memory_session_manager
    if _memory_session_manager is not None:
        return _memory_session_manager
    if not MEMORY_ID:
        return None
    try:
        from bedrock_agentcore.memory.session import MemorySessionManager  # type: ignore

        _memory_session_manager = MemorySessionManager(
            memory_id=MEMORY_ID, region_name=BEDROCK_REGION
        )
        logger.info("AgentCore Memory client ready (memory_id=%s)", MEMORY_ID)
    except Exception as exc:  # pragma: no cover - best-effort only
        logger.warning("Could not initialize AgentCore Memory client: %s", exc)
        _memory_session_manager = None
    return _memory_session_manager


def _persist_turns_to_agentcore_memory(
    user_text: str, ai_text: str, session_id: str, actor_id: str
) -> None:
    """Write the user prompt + assistant response to AgentCore Memory.

    This call is non-blocking on the LLM result: any failure is logged and
    swallowed so memory issues never break a user invocation.
    """
    mgr = _get_memory_session_manager()
    if mgr is None:
        return
    try:
        from bedrock_agentcore.memory.constants import (  # type: ignore
            ConversationalMessage,
            MessageRole,
        )

        session = mgr.create_memory_session(actor_id=actor_id, session_id=session_id)
        session.add_turns(
            messages=[
                ConversationalMessage(user_text, MessageRole.USER),
                ConversationalMessage(ai_text, MessageRole.ASSISTANT),
            ]
        )
        logger.info(
            "Persisted turn to AgentCore Memory (actor=%s session=%s chars_user=%d chars_ai=%d)",
            actor_id,
            session_id,
            len(user_text),
            len(ai_text),
        )
    except Exception as exc:  # pragma: no cover - best-effort only
        logger.warning("Failed to persist turn to AgentCore Memory: %s", exc)


# ---------------------------------------------------------------------------
# LangGraph state + graph
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "conversation history"]


def _extract_text(content) -> str:
    """Normalize an LLM response into a plain string.

    Bedrock Converse (and `langchain-aws`'s `init_chat_model(model_provider="bedrock_converse")`)
    can return `AIMessage.content` either as a plain string or as a list of
    content blocks like `[{"type": "text", "text": "..."}, {"type": "reasoning_content", ...}]`.
    Some models (e.g., `openai.gpt-oss-120b-1:0`) emit reasoning blocks alongside
    or instead of text — so we collect every textual piece we can find and
    join them in order.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Common shapes:
                #   {"type": "text", "text": "..."}
                #   {"type": "reasoning_content", "reasoning_content": {"text": "..."}}
                #   {"type": "tool_use", ...}     -> skipped
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "reasoning_content" in item:
                    rc = item["reasoning_content"]
                    if isinstance(rc, dict) and "text" in rc:
                        parts.append(str(rc["text"]))
                    elif isinstance(rc, str):
                        parts.append(rc)
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
        return "\n".join(p for p in parts if p)
    return str(content)


def _call_model(state: AgentState) -> AgentState:
    """Single graph node: prepend the system prompt and invoke the model."""
    messages: List[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    logger.info("Invoking model with %d messages", len(messages))
    response: AIMessage = llm.invoke(messages)
    return {"messages": state["messages"] + [response]}


def _build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("model", _call_model)
    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)
    return workflow


# Compile once with an in-process MemorySaver. The microVM scope of an AgentCore
# session means in-session memory naturally survives multi-turn calls within a
# session and is discarded when the session ends. AgentCore Memory (above) is
# the layer that persists turns across sessions.
_checkpointer = MemorySaver()
_graph = _build_graph().compile(checkpointer=_checkpointer)

# ---------------------------------------------------------------------------
# AgentCore Runtime entrypoint
# ---------------------------------------------------------------------------
app = BedrockAgentCoreApp()


def _resolve_thread_id(payload: dict, context) -> str:
    """Pick a stable thread id for the LangGraph checkpointer.

    Priority: explicit payload.thread_id -> AgentCore session id -> "default".
    """
    explicit = payload.get("thread_id") if isinstance(payload, dict) else None
    if explicit:
        return str(explicit)
    session_id = getattr(context, "session_id", None) if context is not None else None
    return str(session_id) if session_id else "default"


def _resolve_actor_id(payload: dict) -> str:
    """Pick the actor (user) id for AgentCore Memory.

    Priority: explicit payload.actor_id -> DEFAULT_ACTOR_ID env var -> "research-analyst-user".
    In a real deployment behind a JWT authorizer, you would extract this from the
    validated JWT claims (e.g., `sub` or `username`) — kept simple here.
    """
    if isinstance(payload, dict):
        explicit = payload.get("actor_id") or payload.get("user_id")
        if explicit:
            return str(explicit)
    return DEFAULT_ACTOR_ID


@app.entrypoint
def agent_invocation(payload, context):
    """AgentCore HTTP entrypoint.

    Expected payload: {"prompt": "<user message>",
                       "thread_id": "<optional>",
                       "actor_id":  "<optional>"}
    Returns        : {"result": "<agent reply>", "thread_id": "...", "actor_id": "..."}
    """
    prompt = (payload or {}).get("prompt", "")
    if not prompt:
        return {"result": "Please provide a 'prompt' field in the payload."}

    thread_id = _resolve_thread_id(payload, context)
    actor_id = _resolve_actor_id(payload)
    config = {"configurable": {"thread_id": thread_id}}

    logger.info(
        "Handling invocation thread_id=%s actor_id=%s prompt_chars=%d",
        thread_id,
        actor_id,
        len(prompt),
    )

    result = _graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    raw_answer = result["messages"][-1].content
    answer = _extract_text(raw_answer)

    if not answer:
        logger.warning(
            "Empty answer extracted from model response (raw_type=%s raw_repr=%r)",
            type(raw_answer).__name__,
            raw_answer if not isinstance(raw_answer, list) else f"<list len={len(raw_answer)}>",
        )

    # Persist the turn to AgentCore Memory (write-only — no retrieval injection)
    _persist_turns_to_agentcore_memory(
        user_text=prompt,
        ai_text=answer,
        session_id=thread_id,
        actor_id=actor_id,
    )

    return {"result": answer, "thread_id": thread_id, "actor_id": actor_id}


if __name__ == "__main__":
    app.run()