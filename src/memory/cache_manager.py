import json
import time
from typing import List, Dict, Optional
import os
import redis

from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException
from src.utils.model_loader import ModelLoader
from dotenv import load_dotenv
load_dotenv()

class RedisMemoryManager:
    """
    Production-grade Redis-based session memory manager.

    Supports:
    - Sliding window conversation memory
    - Auto summarization after threshold
    - TTL-based session expiry
    """

    def __init__(
        self,
        redis_url: str= os.getenv("REDIS_URL"),
        ttl_seconds: int = 86400,  # 24 hours
        max_messages: int = 8,
        summarize_threshold: int = 12,
    ):
        try:
            self.redis_client = redis.from_url(redis_url,
                                                decode_responses=True,
                                                socket_timeout=5,
                                                socket_connect_timeout=5,
                                                retry_on_timeout=True,
                                                )
            self.ttl = ttl_seconds
            self.max_messages = max_messages
            self.summarize_threshold = summarize_threshold

            self.llm = ModelLoader().load_llm()

            log.info("RedisMemoryManager initialized")

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing RedisMemoryManager", e
            )

    # =====================================================
    # REDIS KEY
    # =====================================================

    def _key(self, session_id: str) -> str:
        return f"memory:{session_id}"

    # =====================================================
    # LOAD MEMORY
    # =====================================================

    def load(self, session_id: str) -> Dict:
        try:
            raw = self.redis_client.get(self._key(session_id))

            if not raw:
                return {
                    "messages": [],
                    "summary": "",
                }

            data = json.loads(raw)
            return data

        except Exception as e:
            raise ProductAssistantException("Memory load failed", e)

    # =====================================================
    # SAVE MEMORY
    # =====================================================

    def _save(self, session_id: str, data: Dict) -> None:
        try:
            self.redis_client.setex(
                self._key(session_id),
                self.ttl,
                json.dumps(data),
            )
        except Exception as e:
            raise ProductAssistantException("Memory save failed", e)

    # =====================================================
    # APPEND NEW MESSAGE
    # =====================================================

    def append(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        try:
            memory = self.load(session_id)

            messages = memory.get("messages", [])

            messages.append({"role": "user", "content": user_message})
            messages.append({"role": "assistant", "content": assistant_message})

            memory["messages"] = messages

            # Summarize if needed
            if len(messages) >= self.summarize_threshold:
                memory = self._summarize(memory)

            # Apply sliding window
            memory["messages"] = memory["messages"][-self.max_messages :]

            self._save(session_id, memory)

            log.info(
                "memory_updated",
                session_id=session_id,
                message_count=len(memory["messages"]),
                summary_used=bool(memory.get("summary")),
            )

        except Exception as e:
            raise ProductAssistantException("Memory append failed", e)

    # =====================================================
    # SUMMARIZATION
    # =====================================================

    def _summarize(self, memory: Dict) -> Dict:
        try:
            messages = memory.get("messages", [])
            existing_summary = memory.get("summary", "")

            conversation_text = ""

            if existing_summary:
                conversation_text += f"Previous Summary:\n{existing_summary}\n\n"

            for msg in messages:
                conversation_text += f"{msg['role']}: {msg['content']}\n"

            prompt = f"""
Summarize the following conversation briefly while preserving important context:

{conversation_text}

Summary:
"""

            response = self.llm.invoke(prompt)
            summary_text = response.content.strip()

            # After summarizing, reset message history
            memory["summary"] = summary_text
            memory["messages"] = []

            log.info("memory_summarized")

            return memory

        except Exception as e:
            raise ProductAssistantException("Memory summarization failed", e)

    # =====================================================
    # BUILD CONTEXT FOR PROMPT INJECTION
    # =====================================================

    def build_context(self, session_id: str) -> str:
        """
        Returns formatted conversation memory to inject into prompt.
        """

        memory = self.load(session_id)

        summary = memory.get("summary", "")
        messages = memory.get("messages", [])

        context = ""

        if summary:
            context += f"Conversation Summary:\n{summary}\n\n"

        if messages:
            context += "Recent Conversation:\n"
            for msg in messages:
                context += f"{msg['role']}: {msg['content']}\n"

        return context.strip()