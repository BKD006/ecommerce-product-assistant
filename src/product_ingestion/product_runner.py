"""
Product Ingestion Runner (API-ready)

Responsible for:
- Processing uploaded CSV
- Transforming products into embedding-ready records
- Storing them into Pinecone

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

            self.store = ProductVectorStore(index_name=index_name)

            log.info("ProductIngestionRunner initialized successfully")

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing ProductIngestionRunner", e
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
            file_path: temporary CSV file path (from FastAPI)

        Returns:
            dict: ingestion summary
        """

        try:
            log.info(
                "Starting product ingestion from CSV",
                file_path=file_path,
            )

            if not Path(file_path).exists():
                raise ProductAssistantException(
                    f"CSV file not found: {file_path}"
                )

            processor = ProductCatalogProcessor(file_path)
            products = processor.process()

            if not products:
                raise ProductAssistantException(
                    "No valid products found after processing"
                )

            log.info(
                "Product processing completed",
                product_count=len(products),
            )

            self.store.upsert_products(products)

            stats = self.store.describe_index()

            log.info(
                "Product ingestion completed",
                index_stats=stats or {},
            )

            return {
                "status": "success",
                "ingested_products": len(products),
                "index_stats": stats,
            }

        except ProductAssistantException:
            raise
        except Exception as e:
            log.error("CSV ingestion failed", exc_info=True)
            raise ProductAssistantException(
                "Product CSV ingestion failed", e
            )

    # -------------------------------------------------
    # RE-INGEST SINGLE PRODUCT
    # -------------------------------------------------

    def reingest_product(
        self,
        file_path: str,
        product_id: str,
    ) -> dict:
        """
        Re-ingest single product from CSV file.
        """

        try:
            processor = ProductCatalogProcessor(file_path)
            products = processor.process()

            target = [
                p for p in products if p["id"] == product_id
            ]

            if not target:
                raise ProductAssistantException(
                    f"Product {product_id} not found"
                )

            self.store.delete_by_product_id(product_id)
            self.store.upsert_products(target)

            return {
                "status": "success",
                "product_id": product_id,
            }

        except Exception as e:
            raise ProductAssistantException(
                "Re-ingestion failed", e
            )
