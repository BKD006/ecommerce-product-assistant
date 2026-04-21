from pymongo import MongoClient
from datetime import datetime
from typing import List, Dict, Optional


class MongoConversationManager:

    def __init__(self, mongo_uri: str, db_name: str = "rag_system"):

        #Initialize client FIRST
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]

        # Collections
        self.sessions = self.db["sessions"]
        self.messages = self.db["messages"]
        self.summaries = self.db["summaries"]

        # Evaluation collection
        self.evaluations = self.db["evaluations"]

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

    # --------------------------------------------------
    # Save Evaluation
    # --------------------------------------------------

    def save_evaluation(
        self,
        session_id: str,
        evaluation: Dict
    ):
        doc = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            **evaluation
        }

        self.evaluations.insert_one(doc)

    # --------------------------------------------------
    # DEBUG / ANALYTICS METHODS (NEW)
    # --------------------------------------------------

    def get_bad_responses(self, limit: int = 20):
        return list(
            self.evaluations
            .find({"is_bad": True})
            .sort("timestamp", -1)
            .limit(limit)
        )

    def get_failure_stats(self):
        total = self.evaluations.count_documents({})
        bad = self.evaluations.count_documents({"is_bad": True})

        return {
            "total": total,
            "bad": bad,
            "bad_rate": round(bad / total, 3) if total > 0 else 0
        }

    def get_top_failing_queries(self, limit: int = 10):
        pipeline = [
            {"$match": {"is_bad": True}},
            {
                "$group": {
                    "_id": "$query",
                    "count": {"$sum": 1},
                    "avg_score": {"$avg": "$grounding_score"}
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]

        return list(self.evaluations.aggregate(pipeline))