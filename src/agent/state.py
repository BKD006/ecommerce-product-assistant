from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """
    State schema for the Novacart Agent (Research Agentic System).

    This state flows between LangGraph nodes.
    """

    # -------------------------------
    # Core Conversation
    # -------------------------------

    user_query: str
    messages: List[BaseMessage]  # conversation history

    # -------------------------------
    # Reasoning + Control
    # -------------------------------

    iteration: int               # current reasoning iteration
    max_iterations: int          # stopping condition
    next_action: Optional[str]   # "product_tool", "policy_tool", "final"

    reasoning: Optional[str]     # LLM thought step

    # -------------------------------
    # Tool Calls
    # -------------------------------

    tool_calls: List[Dict[str, Any]]  # record of tool usage

    # -------------------------------
    # Tool Results (Observations)
    # -------------------------------

    product_results: Optional[List[Dict[str, Any]]]
    policy_results: Optional[List[Dict[str, Any]]]

    # -------------------------------
    # Final Output
    # -------------------------------

    final_answer: Optional[str]