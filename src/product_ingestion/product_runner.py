from pathlib import Path
from src.product_ingestion.product_loader import ProductCatalogProcessor
from src.product_ingestion.product_vectorstore import ProductVectorStore
from src.exception.custom_exception import ProductAssistantException
from src.logger import GLOBAL_LOGGER as log


class ProductIngestionRunner:
    """
    Stateless ingestion orchestrator.
    """

    def __init__(self, index_name: str):
        try:

            log.info(
                "Initializing ProductIngestionRunner",
                index_name=index_name,
            )

            self.store = ProductVectorStore(
                index_name=index_name
            )

            log.info(
                "ProductIngestionRunner initialized successfully"
            )

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing ProductIngestionRunner",
                e,
            )

    # -------------------------------------------------
    # INGEST FROM CSV FILE
    # -------------------------------------------------

    def ingest_csv(self, file_path: str) -> dict:

        try:

            log.info(
                "Starting product ingestion",
                file_path=file_path,
            )

            file = Path(file_path)

            if not file.exists():
                raise ProductAssistantException(
                    f"CSV file not found: {file_path}"
                )

            # =================================================
            # 🔥 STEP 1: UNIFIED PROCESSING
            # =================================================

            processor = ProductCatalogProcessor(file_path)

            df = processor.process_df()   # ✅ SINGLE SOURCE

            if df.empty:
                raise ProductAssistantException(
                    "No valid products found"
                )

            log.info(
                "Unified dataset ready",
                row_count=len(df),
            )

            # =================================================
            # 🔥 STEP 2: CONVERT DF → VECTOR FORMAT
            # =================================================

            df = processor._build_embedding_text(df)
            df = processor._build_metadata(df)

            products = [
                {
                    "id": str(row["pid"]),
                    "embedding_text": row["embedding_text"],
                    "metadata": row["metadata"],
                }
                for _, row in df.iterrows()
            ]

            log.info(
                "Vector transformation completed",
                product_count=len(products),
            )

            # =================================================
            # 🔥 STEP 3: UPSERT INTO PINECONE
            # =================================================

            self.store.upsert_products(products)

            stats = self.store.describe_index()

            vector_count = None
            dimension = None

            if isinstance(stats, dict):
                vector_count = stats.get("total_vector_count")
                dimension = stats.get("dimension")

            log.info(
                "Product ingestion completed",
                vector_count=vector_count,
                dimension=dimension,
            )

            # =================================================
            # CLEANUP
            # =================================================

            self._delete_file(file)

            return {
                "status": "success",
                "vector_count": vector_count,
                "dimension": dimension,
                "ingested_rows": len(products),  # 🔥 NEW
            }

        except ProductAssistantException:
            raise

        except Exception as e:

            log.error(
                "CSV ingestion failed",
                exc_info=True,
            )

            self._safe_delete(file_path)

            raise ProductAssistantException(
                "Product CSV ingestion failed",
                e,
            )

    # -------------------------------------------------
    # DELETE FILE
    # -------------------------------------------------

    def _delete_file(self, file: Path):

        try:

            if file.exists():

                log.info(
                    "Deleting uploaded file",
                    file=str(file),
                )

                file.unlink()

        except Exception as e:

            log.warning(
                "File deletion failed",
                file=str(file),
                error=str(e),
            )

    # -------------------------------------------------
    # SAFE DELETE
    # -------------------------------------------------

    def _safe_delete(self, file_path: str):

        try:

            file = Path(file_path)

            if file.exists():
                file.unlink()

        except Exception:
            pass