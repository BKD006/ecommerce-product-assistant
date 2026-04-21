from datetime import datetime
from typing import Dict, Optional

from cachetools import TTLCache

from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException
from src.utils.model_loader import ModelLoader


class InMemoryCacheManager:
    """
    In-memory memory using cachetools.TTLCache
    """

    def __init__(
        self,
        ttl_seconds=3600,
        max_sessions=1000,
        max_messages=10,
        summarize_every=10,
    ):
        try:
            # TTL + max size cache
            self.cache = TTLCache(
                maxsize=max_sessions,
                ttl=ttl_seconds,
            )

            self.max_messages = max_messages
            self.summarize_every = summarize_every

            self.llm = ModelLoader().load_llm()

            log.info("InMemoryCacheManager initialized")

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing InMemoryCacheManager",
                e,
            )

    # =====================================================
    # LOAD
    # =====================================================

    def load(self, session_id: str) -> Dict:
        try:
            return self.cache.get(
                session_id,
                {
                    "messages": [],
                    "summary": "",
                    "updated_at": "",
                },
            )

        except Exception as e:
            log.error("memory_load_failed", exc_info=True)
            raise ProductAssistantException("Memory load failed", e)

    # =====================================================
    # SAVE
    # =====================================================

    def _save(self, session_id: str, data: Dict):
        try:
            self.cache[session_id] = data

        except Exception as e:
            log.error("memory_save_failed", exc_info=True)
            raise ProductAssistantException("Memory save failed", e)

    # =====================================================
    # APPEND
    # =====================================================

    def append(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        tool_used: Optional[str] = None,
        citations: Optional[Dict] = None,
    ):
        try:
            memory = self.load(session_id)
            messages = memory.get("messages", [])

            now = datetime.utcnow().isoformat()

            # user
            messages.append({
                "role": "user",
                "content": user_message,
                "time": now,
                "meta": {},
            })

            # assistant
            messages.append({
                "role": "assistant",
                "content": assistant_message,
                "time": now,
                "meta": {
                    "tool": tool_used,
                    "citations": citations or {},
                },
            })

            memory["messages"] = messages

            # summarization trigger
            if len(messages) > self.summarize_every:
                memory = self._summarize(memory)

            # keep last N
            memory["messages"] = memory["messages"][-self.max_messages:]

            memory["updated_at"] = now

            self._save(session_id, memory)

            log.info(
                "memory_updated",
                session_id=session_id,
                message_count=len(memory["messages"]),
                summary_used=bool(memory.get("summary")),
            )

        except Exception as e:
            log.error("memory_append_failed", exc_info=True)
            raise ProductAssistantException("Memory append failed", e)

    # =====================================================
    # SUMMARIZE
    # =====================================================

    def _summarize(self, memory: Dict):
        try:
            messages = memory.get("messages", [])
            summary = memory.get("summary", "")

            if len(messages) <= self.max_messages:
                return memory

            old = messages[:-self.max_messages]
            recent = messages[-self.max_messages:]

            text = ""

            if summary:
                text += f"Previous summary:\n{summary}\n\n"

            for m in old:
                text += f"{m['role']}: {m['content']}\n"

            prompt = f"""
                        Summarize the conversation briefly.
                        Keep important facts, user intent, and decisions.

                        {text}

                        Summary:
                    """

            res = self.llm.invoke(prompt)

            memory["summary"] = res.content.strip()
            memory["messages"] = recent

            log.info(
                "memory_summarized",
                old_count=len(old),
                kept=len(recent),
            )

            return memory

        except Exception as e:
            log.error("memory_summarize_failed", exc_info=True)
            raise ProductAssistantException("Memory summarization failed", e)

    # =====================================================
    # CONTEXT
    # =====================================================

    def build_context(self, session_id: str) -> str:
        try:
            memory = self.load(session_id)

            summary = memory.get("summary", "")
            messages = memory.get("messages", [])

            context = ""

            if summary:
                context += f"Conversation Summary:\n{summary}\n\n"

            if messages:
                context += "Recent Conversation:\n"
                for m in messages:
                    context += f"{m['role']}: {m['content']}\n"

            return context.strip()

        except Exception as e:
            log.error("memory_context_failed", exc_info=True)
            raise ProductAssistantException("Memory context build failed", e)