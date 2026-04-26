import time
import uuid
import os
from typing import Optional

from langsmith import traceable
from langsmith.run_helpers import trace, get_current_run_tree

from src.agent.graph import NovacartAgent
from src.utils.model_loader import ModelLoader
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException
from src.evaluation.rag_eval import RAGEvaluator
from src.memory.cache_manager import InMemoryCacheManager
from src.memory.mongodb_memory_manager import MongoConversationManager
from dotenv import load_dotenv
load_dotenv()

class ObservedAgent:

    def __init__(
        self,
        agent: NovacartAgent,
        memory_manager: Optional[InMemoryCacheManager] = None,
    ):

        self.agent = agent
        self.memory_manager = memory_manager or InMemoryCacheManager()

        # Mongo = source of truth
        self.mongo_manager = MongoConversationManager(
            mongo_uri=os.getenv("MONGO_URI"),
            db_name="rag_system",
        )

        loader = ModelLoader()
        self.embedding_model = loader.load_embeddings("policy")
        self.evaluator = RAGEvaluator(embedding_model=self.embedding_model, llm=ModelLoader().load_llm())

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
            # MEMORY LOAD (ONLY InMemory)
            # =================================================
            memory_context = ""

            if session_id:
                with trace("memory_load"):
                    memory_context = self.memory_manager.build_context(session_id)

            enriched_query = query

            if memory_context:
                enriched_query = f"""
                                Previous Conversation:
                                {memory_context}

                                User Query:
                                {query}
                                """

            # =================================================
            # INITIAL STATE
            # =================================================
            state = {
                "request_id": request_id,
                "user_query": enriched_query,
                "raw_query": query,
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

            # =================================================
            # RAG EVALUATION (FULL)
            # =================================================
            evaluation = {}

            try:
                with trace("rag_evaluation"):
                    evaluation = self.evaluator.evaluate(
                        state=state,
                        query=query,
                        answer=answer,
                        use_llm=True  
                    )

                    # BAD RESPONSE FLAG
                    evaluation["is_bad"] = (
                        evaluation.get("grounding_score", 0) < 0.2
                        or evaluation.get("semantic_similarity", 0) < 0.5
                        or evaluation.get("llm_score", 0) < 0.5
                    )

                    # RESPONSE QUALITY LOG
                    log.info(
                        "response_quality",
                        request_id=request_id,
                        is_bad=evaluation["is_bad"],
                        grounding_score=evaluation.get("grounding_score"),
                        semantic_similarity=evaluation.get("semantic_similarity"),
                        llm_score=evaluation.get("llm_score"),
                    )

                    # ADD: BAD RESPONSE ALERT
                    if evaluation["is_bad"]:
                        log.warning(
                            "bad_response_detected",
                            request_id=request_id,
                            query=query,
                        )

                    log.info(
                        "rag_evaluation_completed",
                        grounding_score=evaluation.get("grounding_score"),
                        hallucination_flag=evaluation.get("hallucination_flag"),
                        semantic_similarity=evaluation.get("semantic_similarity"),
                        llm_score=evaluation.get("llm_score"),
                        llm_verdict=evaluation.get("llm_verdict"),
                    )

            except Exception:
                log.warning(
                    "rag_evaluation_failed",
                    exc_info=True
                )
                evaluation = {}

            total_latency = time.time() - start_total

            # =================================================
            # MEMORY SAVE
            # =================================================
            if session_id:
                with trace("memory_update"):

                    self.mongo_manager.save_message(
                        session_id,
                        "user",
                        {
                            "request_id": request_id,
                            "query": query
                        }
                    )

                    self.mongo_manager.save_message(
                        session_id,
                        "assistant",
                        {
                            "request_id": request_id,
                            "answer": answer
                        }
                    )

                    if evaluation:
                        try:
                            self.mongo_manager.save_evaluation(
                                    session_id=session_id,
                                    evaluation={
                                        "request_id": request_id,
                                        "query": query,
                                        "answer": answer,
                                        **evaluation
                                    }
                                )
                        except Exception:
                            log.warning("Evaluation_storage_failed", exc_info=True)

                    self.memory_manager.append(
                        session_id=session_id,
                        user_message=query,
                        assistant_message=answer,
                    )

            # =================================================
            # LANGSMITH METADATA
            # =================================================
            current_run = get_current_run_tree()

            if current_run:
                current_run.metadata.update({
                    "request_id": request_id,
                    "tool_used": tool_name,
                    "latency": round(total_latency, 4),
                    "session_id": session_id,
                    "evaluation": evaluation
                })

            log.info(
                "agent_request_completed",
                request_id=request_id,
                latency=round(total_latency, 4),
                tool_used=tool_name,
                status="success",
            )

            return {
                "answer": answer,
                "citations": state.get("citations"),
                "evaluation": evaluation
            }

        except Exception as e:
            log.error(
                "agent_failed",
                request_id=request_id,
                error=str(e),
            )
            raise ProductAssistantException("Agent failed", e)