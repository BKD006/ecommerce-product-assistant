from fastapi import APIRouter, HTTPException
from src.schemas.state import ChatRequest, ChatResponse
import uuid
import os

from src.observability.agent_observability import ObservedAgent
from src.agent.graph import NovacartAgent
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException
from src.memory.mongodb_memory_manager import MongoConversationManager

router = APIRouter()

# -------------------------------------------------
# Initialize Agent Once
# -------------------------------------------------

base_agent = NovacartAgent(max_iterations=3)
agent = ObservedAgent(base_agent)

# -------------------------------------------------
# Mongo (only for session creation)
# -------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI")
mongo_memory = MongoConversationManager(MONGO_URI)


# -------------------------------------------------
# Chat Endpoint
# -------------------------------------------------

@router.post("/query", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        log.info(
            "Chat request received",
            query=request.query,
            session_id=request.session_id,
        )

        if not request.query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty",
            )

        session_id = request.session_id or str(uuid.uuid4())

        # -----------------------------
        # Create session only once
        # -----------------------------

        if request.session_id is None:

            mongo_memory.create_session(
                session_id=session_id,
                user_id="anonymous",
            )

        # -----------------------------
        # Run agent
        # -----------------------------

        result = agent.run(
            query=request.query,
            session_id=session_id,
        )

        if isinstance(result, dict):
            answer = result.get("answer", "")
            citations = result.get("citations", {})
        else:
            answer = result
            citations = None

        return ChatResponse(
            status="success",
            answer=answer,
            session_id=session_id,
            citations=citations,
        )

    except ProductAssistantException as e:

        log.error(
            "Agent execution failed",
            error=str(e),
        )

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except Exception:

        log.error(
            "Unexpected error in chat endpoint",
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )