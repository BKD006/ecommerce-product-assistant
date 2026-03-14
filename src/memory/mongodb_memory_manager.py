from pymongo import MongoClient
from datetime import datetime
from typing import List, Dict, Optional


class MongoConversationManager:
    """
    MongoDB-based conversation memory manager.

    Collections:
        - sessions
        - messages
        - summaries
    """

    def __init__(self, mongo_uri: str, db_name: str = "rag_system"):

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]

        self.sessions = self.db["sessions"]
        self.messages = self.db["messages"]
        self.summaries = self.db["summaries"]

        # Index for fast retrieval
        self.messages.create_index([("session_id", 1), ("timestamp", 1)])

    # --------------------------------------------------
    # Session Management
    # --------------------------------------------------

    def create_session(self, session_id: str, user_id: Optional[str] = None):

        session = {
            "_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "last_updated": datetime.utcnow()
        }

        self.sessions.update_one(
            {"_id": session_id},
            {"$setOnInsert": session},
            upsert=True
        )

    # --------------------------------------------------
    # Save Message
    # --------------------------------------------------

    def save_message(self, session_id: str, role: str, content: str):

        message = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }

        self.messages.insert_one(message)

        self.sessions.update_one(
            {"_id": session_id},
            {"$set": {"last_updated": datetime.utcnow()}}
        )

    # --------------------------------------------------
    # Get Conversation History
    # --------------------------------------------------

    def get_conversation(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict]:

        cursor = (
            self.messages
            .find({"session_id": session_id})
            .sort("timestamp", 1)
            .limit(limit)
        )

        return list(cursor)

    # --------------------------------------------------
    # Clear Session
    # --------------------------------------------------

    def delete_session(self, session_id: str):

        self.sessions.delete_one({"_id": session_id})
        self.messages.delete_many({"session_id": session_id})
        self.summaries.delete_many({"session_id": session_id})
