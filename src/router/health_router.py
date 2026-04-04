from fastapi import APIRouter
from datetime import datetime

from src.sql.sqlite_manager import SQLiteManager
from src.product_ingestion.product_vectorstore import ProductVectorStore
from src.policy_ingestion.policy_vectorstore import PolicyVectorStore
from src.utils.config_loader import load_config
from src.logger import GLOBAL_LOGGER as log

router = APIRouter()

@router.get("/")
def health_check():
    """
    Overall system health check
    """

    status = {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    # =================================================
    # SQLite Check
    # =================================================
    try:
        db = SQLiteManager("ecommerce.db")
        db.execute_query("SELECT 1")
        status["services"]["sqlite"] = "ok"
    except Exception as e:
        log.error("SQLite health check failed", error=str(e))
        status["services"]["sqlite"] = "down"
        status["status"] = "degraded"

    # =================================================
    # Pinecone Check (Product Vector DB)
    # =================================================
    try:
        config = load_config()
        index_name = config["product_index"]["name"]

        store = ProductVectorStore(index_name=index_name)
        stats = store.describe_index()

        status["services"]["pinecone"] = {
            "status": "ok",
            "vector_count": stats.get("total_vector_count")
        }

    except Exception as e:
        log.error("Pinecone health check failed", error=str(e))
        status["services"]["pinecone"] = "down"
        status["status"] = "degraded"

    # =================================================
    # Qdrant Check (Policy DB)
    # =================================================
    try:
        collection_name = config.get(
            "policy_collection",
            "novacart_support_policies"
        )

        policy_store = PolicyVectorStore(
            collection_name=collection_name
        )

        count = policy_store.count()

        status["services"]["qdrant"] = {
            "status": "ok",
            "vector_count": count
        }

    except Exception as e:
        log.error("Qdrant health check failed", error=str(e))
        status["services"]["qdrant"] = "down"
        status["status"] = "degraded"

    return status