from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from agent.tools import validate_and_run_tool


MAX_TOOL_CALLS = 4
MAX_STEPS = 6


class AgentState(TypedDict):
    messages: list[BaseMessage]
    intermediate_results: list[str]
    step_count: int
    tool_call_count: int
    next_action: str
    planned_tool: str
    planned_arguments: dict[str, Any]
    recovery_attempts: int


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
)


# ============================================================
# Pattern 1: Planner / Executor
# ============================================================

def planner_node(state: AgentState) -> AgentState:
    state["step_count"] += 1

    state["intermediate_results"].append(
        "Transition: planner"
    )

    if state["step_count"] > MAX_STEPS:
        state["next_action"] = "fallback"
        state["intermediate_results"].append(
            "Recovery: Maximum reasoning steps reached."
        )
        return state

    user_text = str(state["messages"][0].content)
    text = user_text.lower()

    math_chars = set("0123456789+-*/().% ")

    if (
        any(
            word in text
            for word in ["calculate", "compute", "what is"]
        )
        and any(char.isdigit() for char in text)
    ):
        expression = "".join(
            char for char in user_text
            if char in math_chars
        ).strip()

        state["planned_tool"] = "calculator"
        state["planned_arguments"] = {
            "expression": expression
        }

    elif any(
        phrase in text
        for phrase in [
            "save this",
            "save a note",
            "write this down",
            "remember this note",
        ]
    ):
        state["planned_tool"] = "save_note"
        state["planned_arguments"] = {
            "note": user_text
        }

    else:
        state["planned_tool"] = "web_search"
        state["planned_arguments"] = {
            "query": user_text
        }

    state["intermediate_results"].append(
        "Planner selected: "
        f"{state['planned_tool']} "
        f"with arguments {state['planned_arguments']}"
    )

    state["next_action"] = "tools"
    return state


# ============================================================
# Pattern 2: ReAct
# ============================================================

def react_node(state: AgentState) -> AgentState:
    state["step_count"] += 1

    state["intermediate_results"].append(
        "Transition: react"
    )

    if state["step_count"] > MAX_STEPS:
        state["next_action"] = "fallback"
        state["intermediate_results"].append(
            "Recovery: ReAct reached maximum steps."
        )
        return state

    state["intermediate_results"].append(
        "ReAct: inspect tool result and decide next action."
    )

    # Controlled cycle:
    # ReAct -> Planner -> Tools -> ReAct
    # This only happens after a recovery attempt.
    if (
        state["recovery_attempts"] > 0
        and state["step_count"] < MAX_STEPS
    ):
        state["next_action"] = "planner"
    else:
        state["next_action"] = "respond"

    return state


# ============================================================
# Tool Executor
# ============================================================

def tools_node(state: AgentState) -> AgentState:
    state["intermediate_results"].append(
        "Transition: tools"
    )

    if state["tool_call_count"] >= MAX_TOOL_CALLS:
        state["intermediate_results"].append(
            "Recovery: Tool-call limit reached. "
            "Falling back safely."
        )
        state["next_action"] = "fallback"
        return state

    tool_name = state["planned_tool"]
    arguments = state["planned_arguments"]

    state["tool_call_count"] += 1

    state["intermediate_results"].append(
        f"Tool call started: {tool_name}"
    )

    result = validate_and_run_tool(
        tool_name,
        arguments,
    )

    failed = (
        result.startswith("Validation error")
        or result.startswith("Tool execution error")
        or result.startswith("Web search failed")
        or result.startswith("Calculator failed")
        or result.startswith("Could not save")
        or result.startswith("Unknown tool")
    )

    if failed:
        state["recovery_attempts"] += 1

        state["intermediate_results"].append(
            f"Recovery: {result}"
        )

        if (
            state["recovery_attempts"] >= 2
            or state["tool_call_count"] >= MAX_TOOL_CALLS
        ):
            state["next_action"] = "fallback"
        else:
            state["next_action"] = "react"

        return state

    state["intermediate_results"].append(
        f"Tool call: {tool_name}\n"
        f"Result: {result}"
    )

    state["next_action"] = "react"
    return state


# ============================================================
# Final Response
# ============================================================

def respond_node(state: AgentState) -> AgentState:
    state["intermediate_results"].append(
        "Transition: respond"
    )

    results = "\n\n".join(
        state["intermediate_results"][-5:]
    )

    prompt = f"""
You are Karim's Second-Brain Agent.

Answer the user's request concisely.

Use the tool results below when relevant.

Do not expose internal tool metadata,
signatures, Python structures, or internal reasoning.

Return only the natural-language answer.

Tool results:
{results}
"""

    try:
        response = llm.invoke(
            state["messages"]
            + [HumanMessage(content=prompt)]
        )

        content = response.content

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(
                            str(item.get("text", ""))
                        )
                else:
                    text_parts.append(str(item))

            content = "\n".join(
                part for part in text_parts if part
            )

        state["messages"].append(
            AIMessage(content=str(content))
        )

    except Exception as exc:
        state["messages"].append(
            AIMessage(
                content=(
                    "I could not generate the final response safely: "
                    f"{exc}"
                )
            )
        )

    return state


# ============================================================
# Safe Fallback
# ============================================================

def fallback_node(state: AgentState) -> AgentState:
    state["intermediate_results"].append(
        "Transition: fallback"
    )

    state["intermediate_results"].append(
        "Fallback: stopped safely without further tool calls."
    )

    state["messages"].append(
        AIMessage(
            content=(
                "I reached a safety limit while processing "
                "this request. No further tool calls were made."
            )
        )
    )

    state["next_action"] = "end"
    return state


# ============================================================
# Routing
# ============================================================

def route_after_planner(state: AgentState) -> str:
    if state["next_action"] == "fallback":
        return "fallback"

    return "tools"


def route_after_tools(state: AgentState) -> str:
    if state["next_action"] == "fallback":
        return "fallback"

    return "react"


# IMPORTANT:
# This is intentionally a NEW router name.
# It avoids the previous duplicate branch-name problem.
def react_router_v2(state: AgentState) -> str:
    if state["next_action"] == "fallback":
        return "fallback"

    if state["next_action"] == "planner":
        return "planner"

    return "respond"


# ============================================================
# LangGraph
# ============================================================

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("tools", tools_node)
builder.add_node("react", react_node)
builder.add_node("respond", respond_node)
builder.add_node("fallback", fallback_node)

builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "tools": "tools",
        "fallback": "fallback",
    },
)

builder.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "react": "react",
        "fallback": "fallback",
    },
)

# Real controlled cycle:
# react -> planner -> tools -> react
builder.add_conditional_edges(
    "react",
    react_router_v2,
    {
        "planner": "planner",
        "respond": "respond",
        "fallback": "fallback",
    },
)

builder.add_edge("respond", END)
builder.add_edge("fallback", END)

graph = builder.compile()


# ============================================================
# Agent Runner + Printed Run Log
# ============================================================

def run_agent(user_input: str) -> AgentState:
    initial_state: AgentState = {
        "messages": [
            HumanMessage(content=user_input)
        ],
        "intermediate_results": [],
        "step_count": 0,
        "tool_call_count": 0,
        "next_action": "planner",
        "planned_tool": "",
        "planned_arguments": {},
        "recovery_attempts": 0,
    }

    print(
        "\n========== KARIM SECOND-BRAIN RUN =========="
    )
    print(f"USER: {user_input}")
    print("--------------------------------------------")

    result = graph.invoke(initial_state)

    print("RUN LOG:")

    for item in result["intermediate_results"]:
        print(item)

    print("--------------------------------------------")
    print(f"Steps: {result['step_count']}")
    print(f"Tool calls: {result['tool_call_count']}")
    print(
        f"Recovery attempts: "
        f"{result['recovery_attempts']}"
    )
    print(
        "============================================\n"
    )

    return result