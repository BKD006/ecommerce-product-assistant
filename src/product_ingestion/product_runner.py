"""
Product Ingestion Runner (API-ready)

Responsible for:
- Processing uploaded CSV
- Transforming products into embedding-ready records
- Storing them into Pinecone
- Deleting uploaded file after ingestion

Designed for FastAPI file-upload usage.
"""

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

    def ingest_csv(
        self,
        file_path: str,
    ) -> dict:
        """
        Ingest products from uploaded CSV file.

        Args:
            file_path: temporary CSV file path

        Returns:
            dict: ingestion summary
        """

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

            # ----------------------------
            # Process CSV
            # ----------------------------

            processor = ProductCatalogProcessor(
                file_path
            )

            products = processor.process()

            if not products:
                raise ProductAssistantException(
                    "No valid products found"
                )

            log.info(
                "Product processing completed",
                product_count=len(products),
            )

            # ----------------------------
            # Store in Pinecone
            # ----------------------------

            self.store.upsert_products(products)

            stats = self.store.describe_index()

            vector_count = None
            dimension = None

            if isinstance(stats, dict):
                vector_count = stats.get(
                    "total_vector_count"
                )
                dimension = stats.get(
                    "dimension"
                )

            log.info(
                "Product ingestion completed",
                vector_count=vector_count,
                dimension=dimension,
            )

            # ----------------------------
            # CLEANUP FILE
            # ----------------------------

            self._delete_file(file)

            return {
                "status": "success",
                "vector_count": vector_count,
                "dimension": dimension,
            }

        except ProductAssistantException:
            raise

        except Exception as e:

            log.error(
                "CSV ingestion failed",
                exc_info=True,
            )

            # delete file even if failed
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