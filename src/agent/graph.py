import json
from typing import Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from src.agent.state import AgentState
from src.agent.prompts import reasoning_prompt, final_prompt
from src.tools.tool_registry import ToolRegistry
from src.utils.model_loader import ModelLoader
from src.logger import GLOBAL_LOGGER as log


class NovacartAgent:
    """
    Production-ready multi-tool ecommerce agent using LangGraph.

    Architecture:
        Reason → (Tool OR Final) → Final → END

    Design Principles:
        - Single tool execution per query
        - No reflection loop (prevents infinite execution)
        - Hard iteration cap
        - Safe JSON parsing
        - Deterministic termination

    LLM Calls Per Query:
        - 1x Reason
        - 1x Final
        Maximum = 2

    This prevents:
        - 429 rate limit loops
        - Repeated tool execution
        - Infinite graph cycles
    """

    def __init__(self, max_iterations: int = 1):
        self.max_iterations = max_iterations

        # Load primary LLM
        self.llm = ModelLoader().load_llm()

        # Tool registry abstraction
        self.tool_registry = ToolRegistry()

        # Build graph
        self.graph = self._build_graph()

    # =================================================
    # PUBLIC RUN METHOD
    # =================================================

    def run(self, user_query: str) -> str:
        """
        Executes the LangGraph agent.

        Args:
            user_query (str): User input query

        Returns:
            str: Final generated response
        """

        initial_state: AgentState = {
            "user_query": user_query,
            "messages": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "next_action": None,
            "reasoning": None,
            "tool_query": None,
            "tool_calls": [],
            "product_results": None,
            "policy_results": None,
            "final_answer": None,
        }

        final_state = self.graph.invoke(initial_state)

        return final_state.get("final_answer", "No response generated.")

    # =================================================
    # REASON NODE
    # =================================================

    def _reason_node(self, state: AgentState) -> AgentState:
        """
        Decides whether to:
            - Call a tool
            - Generate final answer directly

        Enforces hard iteration cap.
        """

        if state["iteration"] >= state["max_iterations"]:
            log.info("Max iterations reached in reason node")
            state["next_action"] = "final"
            return state

        prompt = reasoning_prompt(state["user_query"])

        try:
            response = self.llm.invoke(
                [HumanMessage(content=prompt)]
            )
            decision = json.loads(response.content.strip())

        except Exception:
            log.warning("Reason JSON parsing failed. Defaulting to final.")
            decision = {
                "thought": "Parsing failed",
                "action": "final",
                "tool_query": state["user_query"],
            }

        state["reasoning"] = decision.get("thought", "")
        state["next_action"] = decision.get("action", "final")
        state["tool_query"] = decision.get(
            "tool_query",
            state["user_query"],
        )

        state["iteration"] += 1

        log.info(
            "Reason step completed",
            iteration=state["iteration"],
            action=state["next_action"],
        )

        return state

    # =================================================
    # TOOL NODE
    # =================================================

    def _tool_node(self, state: AgentState) -> AgentState:
        """
        Executes the selected tool once.

        Safety:
            - Prevents repeated identical tool calls
            - Only one tool execution per query
        """

        action = state["next_action"]
        query = state.get("tool_query", state["user_query"])

        # Prevent repeated identical tool calls
        if state["tool_calls"]:
            log.info("Tool already executed once. Skipping re-execution.")
            state["next_action"] = "final"
            return state

        if action in self.tool_registry.list_tools():

            results = self.tool_registry.execute(
                action,
                query=query,
            )

            if action == "product_tool":
                state["product_results"] = results

            elif action == "policy_tool":
                state["policy_results"] = results

        elif action == "both":

            if "product_tool" in self.tool_registry.list_tools():
                state["product_results"] = self.tool_registry.execute(
                    "product_tool",
                    query=query,
                )

            if "policy_tool" in self.tool_registry.list_tools():
                state["policy_results"] = self.tool_registry.execute(
                    "policy_tool",
                    query=query,
                )

        state["tool_calls"].append(
            {"tool": action, "query": query}
        )

        log.info("Tool executed", tool=action)

        return state

    # =================================================
    # FINAL NODE
    # =================================================

    def _final_node(self, state: AgentState) -> AgentState:
        """
        Generates final answer using:
            - User query
            - Retrieved product results
            - Retrieved policy results
        """

        prompt = final_prompt(
            state["user_query"],
            state.get("product_results"),
            state.get("policy_results"),
        )

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        state["final_answer"] = response.content

        log.info("Final answer generated")

        return state

    # =================================================
    # ROUTING LOGIC
    # =================================================

    def _route_after_reason(
        self, state: AgentState
    ) -> Literal["tool", "final"]:
        """
        Routes execution after reasoning.
        """

        if state["next_action"] == "final":
            return "final"

        return "tool"

    # =================================================
    # GRAPH BUILDER
    # =================================================

    def _build_graph(self):
        """
        Constructs LangGraph workflow:

            reason
               ├── tool
               └── final

            tool → final
            final → END
        """

        graph = StateGraph(AgentState)

        graph.add_node("reason", self._reason_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("final", self._final_node)

        graph.set_entry_point("reason")

        # Conditional routing after reason
        graph.add_conditional_edges(
            "reason",
            self._route_after_reason,
            {
                "tool": "tool",
                "final": "final",
            },
        )

        graph.add_edge("tool", "final")
        graph.add_edge("final", END)

        return graph.compile()