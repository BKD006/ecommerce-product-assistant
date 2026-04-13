import json
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from src.schemas.models import AgentState
from src.agent.prompts import final_prompt
from src.tools.tool_registry import ToolRegistry
from src.utils.model_loader import ModelLoader
from src.logger import GLOBAL_LOGGER as log
from src.agent.query_understanding import QueryUnderstanding
from src.utils.config_loader import load_config
from src.retriever.product_retriever import HybridProductRetrieverV2

class NovacartAgent:

    def __init__(self, max_iterations: int = 2):
        self.max_iterations = max_iterations
        self.llm = ModelLoader().load_llm()
        self.tool_registry = ToolRegistry()
        self.graph = self._build_graph()
        config = load_config()
        index_name = config["product_index"]["name"]
        retriever = HybridProductRetrieverV2(index_name=index_name)
        categories = retriever.get_unique_metadata("category")
        brands = retriever.get_unique_metadata("brand")

        self.query_understanding = QueryUnderstanding(
            llm=self.llm,
            known_categories=categories,
            known_brands=brands
        )
    # =================================================
    # PUBLIC RUN (ObservedAgent usually calls nodes)
    # =================================================

    def run(self, user_query: str):

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
            "citations": None,
        }
        final_state = self.graph.invoke(initial_state)
        return {
        "answer": final_state.get("final_answer"),
        "type": final_state.get("type", "text"),
        "data": final_state.get("data", None),
        "citations": final_state.get("citations", {}),
        }

    # =================================================
    # REASON NODE
    # =================================================

    def _reason_node(self, state):

        raw_query = state.get("raw_query", state["user_query"])

        # ---------------------------------------
        # Query Understanding
        # ---------------------------------------
        parsed = self.query_understanding.parse(raw_query)

        log.info("Parsed Query", parsed=parsed)

        intent = parsed.get("intent")

        # ---------------------------------------
        # DETERMINISTIC ROUTING
        # ---------------------------------------

        # 1. POLICY
        if intent == "policy":
            action = "policy_tool"

        # 2. PRODUCT (DEFAULT)
        else:
            action = "product_tool"

        # ---------------------------------------
        # TOOL QUERY
        # ---------------------------------------
        if parsed.get("aggregation_type"):
            tool_query = raw_query   # keep full query for aggregation
        else:
            tool_query = parsed.get("clean_query") or raw_query

        # ---------------------------------------
        # UPDATE STATE
        # ---------------------------------------
        state["next_action"] = action
        state["tool_query"] = tool_query
        state["parsed"] = parsed
        state["iteration"] += 1

        log.info("Deterministic Routing", action=action)

        return state
    # =================================================
    # TOOL NODE
    # =================================================

    def _tool_node(self, state: AgentState) -> AgentState:

        action = state.get("next_action")
        query = state.get("tool_query", state["user_query"])
        parsed = state.get("parsed", {})

        if not action or action == "final":
            return state

        # =====================================================
        # CLEAN FILTERS (NO None VALUES)
        # =====================================================
        raw_filters = {
            "category": parsed.get("category"),
            "brand": parsed.get("brand"),
            "min_price": parsed.get("price_min"),
            "max_price": parsed.get("price_max"),
            "aggregation_type": parsed.get("aggregation_type"),  
        }

        filters = {
            k: v for k, v in raw_filters.items()
            if v is not None
        }

        log.info("Filters Applied", filters=filters)

        try:
            # =====================================================
            # TOOL EXECUTION
            # =====================================================

            if action == "product_tool":
                results = self.tool_registry.execute(
                    "product_tool",
                    query=query,
                    filters=filters
                )
                state["product_results"] = results

            elif action == "policy_tool":
                results = self.tool_registry.execute(
                    "policy_tool",
                    query=query
                )
                state["policy_results"] = results

            else:
                raise ValueError(f"Invalid tool: {action}")

        except Exception:
            log.error("Tool execution failed", exc_info=True)
            state["next_action"] = "final"
            return state

        # =====================================================
        # TRACK TOOL CALLS
        # =====================================================
        state["tool_calls"].append({
            "tool": action,
            "query": query,
            "filters": filters,
        })

        return state

    # =================================================
    # FINAL NODE
    # =================================================

    def _final_node(self, state: AgentState) -> AgentState:

        product_results = state.get("product_results") or []
        policy_results = state.get("policy_results") or []

        # =====================================================
        # SAFETY
        # =====================================================
        if not isinstance(product_results, list):
            product_results = []

        if not isinstance(policy_results, list):
            policy_results = []

        formatted_products = []
        formatted_policies = []
        source_map = {}
        counter = 1

        # -------- products --------
        for p in product_results:
            if not isinstance(p, dict):
                continue

            formatted_products.append(
                                    f"""
                                    Product {counter}:
                                    - Brand: {p.get('brand', 'Unknown')}
                                    - Category: {p.get('category', 'Unknown')}
                                    - Price: ₹{p.get('price', 'N/A')}
                                    - Description: {p.get('content', '')[:150]}
                                    """
                                    )
            source_map[str(counter)] = p
            counter += 1

        # -------- policies --------
        for pol in policy_results:
            if not isinstance(pol, dict):
                continue

            formatted_policies.append(
                f"[{counter}] Policy | "
                f"{pol.get('policy_type')} | "
                f"{pol.get('section_title')}\n"
                f"{pol.get('content')}"
            )
            source_map[str(counter)] = pol
            counter += 1

        # =====================================================
        # LLM GENERATION
        # =====================================================
        if not product_results and not policy_results:
            state["final_answer"] = "I couldn’t find relevant products for your query. Try being more specific (e.g., brand, category, or price range)."
            state["type"] = "text"
            state["data"] = None
            state["citations"] = {}
            return state
        prompt = final_prompt(
            state["user_query"],
            "\n\n".join(formatted_products),
            "\n\n".join(formatted_policies),
        )

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
        )

        state["final_answer"] = response.content
        state["type"] = "text"
        state["data"] = None
        state["citations"] = source_map

        return state

    # =================================================
    # ROUTER
    # =================================================

    def _route_after_reason(
        self,
        state: AgentState,
    ) -> Literal["tool", "final"]:
        if state["next_action"] == "final":
            return "final"
        return "tool"

    # =================================================
    # GRAPH
    # =================================================

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("reason", self._reason_node)
        graph.add_node("tool", self._tool_node)
        graph.add_node("final", self._final_node)
        graph.set_entry_point("reason")
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