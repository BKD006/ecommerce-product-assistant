from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

class AgentState(TypedDict, total=False):

    # ===============================
    # INPUT
    # ===============================

    user_query: str

    # ===============================
    # CHAT HISTORY (optional)
    # ===============================

    messages: List[BaseMessage]

    # ===============================
    # LOOP CONTROL
    # ===============================

    iteration: int
    max_iterations: int

    next_action: Optional[str]
    reasoning: Optional[str]

    tool_query: Optional[str]

    # ===============================
    # TOOL CALLS
    # ===============================

    tool_calls: List[Dict[str, Any]]

    # ===============================
    # TOOL RESULTS
    # ===============================

    product_results: Optional[List[Dict[str, Any]]]
    policy_results: Optional[List[Dict[str, Any]]]

    # ===============================
    # FINAL OUTPUT
    # ===============================

    final_answer: Optional[str]

    citations: Optional[Dict[str, Any]]

    # ===============================
    # TRACEABILITY
    # ===============================
    request_id: Optional[str]

# ===============================
# FastAPI Schemas
# ===============================

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    status: str
    answer: str
    session_id: str
    citations: Optional[Dict[str, Any]] = None
    type: Optional[str] = None
    data: Optional[Any] = None