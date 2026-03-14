import time
import uuid
import os
from typing import Dict, Any, Optional
import numpy as np

from langsmith import traceable
from langsmith.run_helpers import trace, get_current_run_tree

from src.agent.graph import NovacartAgent
from src.utils.model_loader import ModelLoader
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException

from src.memory.redis_memory_manager import RedisMemoryManager
from src.memory.mongodb_memory_manager import MongoConversationManager
from dotenv import load_dotenv
load_dotenv()

class ObservedAgent:

    def __init__(
        self,
        agent: NovacartAgent,
        memory_manager: Optional[RedisMemoryManager] = None,
    ):

        self.agent = agent
        self.memory_manager = memory_manager or RedisMemoryManager()

        # Mongo = source of truth
        self.mongo_manager = MongoConversationManager(
            mongo_uri=os.getenv("MONGO_URI"),
            db_name="rag_system",
        )

        loader = ModelLoader()
        self.embedding_model = loader.load_embeddings("policy")

    # =====================================================
    # HALLUCINATION CHECK
    # =====================================================

    def _hallucination_check(
        self,
        state: Dict[str, Any],
        answer: str,
        memory_message_count: int,
    ):

        product_results = state.get("product_results") or []
        policy_results = state.get("policy_results") or []

        retrieved_text = ""

        for r in product_results:
            retrieved_text += " " + (r.get("content") or "")

        for r in policy_results:
            retrieved_text += " " + (r.get("content") or "")

        # allow memory answers
        if not retrieved_text.strip():

            if memory_message_count > 0:
                return {
                    "risk_score": 0.2,
                    "semantic_similarity": 0.0,
                    "flag": False,
                }

            return {
                "risk_score": 0.9,
                "semantic_similarity": 0.0,
                "flag": True,
            }

        try:

            answer_vector = np.array(
                self.embedding_model.embed_query(answer)
            )

            context_vector = np.array(
                self.embedding_model.embed_query(retrieved_text)
            )

            similarity = np.dot(answer_vector, context_vector) / (
                np.linalg.norm(answer_vector)
                * np.linalg.norm(context_vector)
            )

            similarity = float(similarity)

            if similarity > 0.80:
                risk_score = 0.1
            elif similarity > 0.65:
                risk_score = 0.4
            elif similarity > 0.50:
                risk_score = 0.7
            else:
                risk_score = 0.9

            return {
                "risk_score": round(risk_score, 3),
                "semantic_similarity": round(similarity, 3),
                "flag": risk_score >= 0.6,
            }

        except Exception:

            return {
                "risk_score": 0.5,
                "semantic_similarity": 0.0,
                "flag": False,
            }

    # =====================================================
    # MAIN RUN
    # =====================================================

    @traceable(name="ObservedAgentRun")
    def run(self, query: str, session_id: Optional[str] = None):

        request_id = str(uuid.uuid4())
        start_total = time.time()

        log.info(
            "agent_request_started",
            request_id=request_id,
            query=query,
            session_id=session_id,
        )

        try:

            # =================================================
            # MEMORY LOAD
            # =================================================

            memory_context = ""
            memory_message_count = 0
            summary_used = False

            if session_id:

                with trace("memory_load"):

                    memory_data = self.memory_manager.load(session_id)

                    # fallback to mongo
                    if (
                        not memory_data["messages"]
                        and not memory_data["summary"]
                    ):

                        mongo_messages = self.mongo_manager.get_conversation(
                            session_id
                        )

                        last_user = None

                        for m in mongo_messages:

                            if m["role"] == "user":
                                last_user = m["content"]

                            else:
                                if last_user:
                                    self.memory_manager.append(
                                        session_id,
                                        last_user,
                                        m["content"],
                                    )

                        memory_data = self.memory_manager.load(session_id)

                    memory_context = self.memory_manager.build_context(
                        session_id
                    )

                    memory_message_count = len(
                        memory_data.get("messages", [])
                    )

                    summary_used = bool(
                        memory_data.get("summary")
                    )

            enriched_query = query

            if memory_context:

                enriched_query = f"""
Previous Conversation Context:
{memory_context}

Current User Query:
{query}
"""

            # =================================================
            # INITIAL STATE
            # =================================================

            state = {
                "user_query": enriched_query,
                "messages": [],
                "iteration": 0,
                "max_iterations": self.agent.max_iterations,
                "next_action": None,
                "reasoning": None,
                "tool_query": None,
                "tool_calls": [],
                "product_results": None,
                "policy_results": None,
                "final_answer": None,
                "citations": None,
            }

            # =================================================
            # REASON
            # =================================================

            with trace("reason_stage"):
                state = self.agent._reason_node(state)

            tool_name = state.get("next_action")

            # =================================================
            # TOOL
            # =================================================

            if tool_name != "final":

                with trace("tool_stage"):
                    state = self.agent._tool_node(state)

            # =================================================
            # FINAL
            # =================================================

            with trace("final_stage"):
                state = self.agent._final_node(state)

            answer = state.get("final_answer", "")

            total_latency = time.time() - start_total

            # =================================================
            # MEMORY SAVE
            # =================================================

            if session_id:

                with trace("memory_update"):

                    self.mongo_manager.save_message(
                        session_id,
                        "user",
                        query,
                    )

                    self.mongo_manager.save_message(
                        session_id,
                        "assistant",
                        answer,
                    )

                    self.memory_manager.append(
                        session_id=session_id,
                        user_message=query,
                        assistant_message=answer,
                    )

            # =================================================
            # HALLUCINATION CHECK
            # =================================================

            hallucination_metrics = self._hallucination_check(
                state,
                answer,
                memory_message_count,
            )

            current_run = get_current_run_tree()

            if current_run:

                current_run.metadata.update({
                    "request_id": request_id,
                    "tool_used": tool_name,
                    "total_latency": round(total_latency, 4),
                    "session_id": session_id,
                    "memory_message_count": memory_message_count,
                    "memory_summary_used": summary_used,
                    "hallucination_flag": hallucination_metrics["flag"],
                    "hallucination_risk_score": hallucination_metrics["risk_score"],
                })

            log.info(
                "agent_request_completed",
                request_id=request_id,
                total_latency=round(total_latency, 4),
                tool_used=tool_name,
                hallucination_flag=hallucination_metrics["flag"],
                memory_message_count=memory_message_count,
                status="success",
            )

            return {
                "answer": answer,
                "citations": state.get("citations"),
            }

        except ProductAssistantException as e:

            log.error(
                "agent_request_failed",
                request_id=request_id,
                error=str(e),
            )

            raise

        except Exception as e:

            log.error(
                "agent_unexpected_error",
                request_id=request_id,
                error=str(e),
            )

            raise