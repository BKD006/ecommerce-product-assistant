import math
from typing import List, Tuple

from src.sql.sqlite_manager import SQLiteManager
from src.product_ingestion.product_loader import ProductCatalogProcessor
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException


class SQLiteIngestion:
    """
    Handles CSV → SQLite ingestion.
    Uses ProductCatalogProcessor as single source of truth.
    """

    def __init__(self, batch_size: int = 500):
        self.db = SQLiteManager("ecommerce.db")
        self.batch_size = batch_size

    # =====================================================
    # PUBLIC METHOD
    # =====================================================

    def ingest_csv(self, file_path: str):

        try:
            log.info("Starting SQLite ingestion", file=file_path)

            # Use Unified ProductCatalogProcessor for consistent processing
            processor = ProductCatalogProcessor(file_path)
            df = processor.process_df()

            if df.empty:
                raise ProductAssistantException(
                    "No valid data after processing"
                )

            total_rows = len(df)

            log.info(
                "Processed dataset ready for SQLite",
                rows=total_rows
            )

            # Direct insert (no extra cleaning, dedup)
            self._batch_insert(df)

            log.info(
                "SQLite ingestion completed successfully",
                total_rows=total_rows,
            )

        except Exception as e:
            log.error("SQLite ingestion failed", exc_info=True)
            raise ProductAssistantException(
                "SQLite ingestion failed", e
            )

    # =====================================================
    # BATCH INSERT
    # =====================================================

    def _batch_insert(self, df):

        total_rows = len(df)
        batches = math.ceil(total_rows / self.batch_size)

        log.info(
            "Starting batch insertion",
            total_rows=total_rows,
            batch_size=self.batch_size,
            batches=batches,
        )

        insert_query = """
        INSERT OR IGNORE INTO products (
            product_id,
            product_name,
            brand,
            category,
            sub_category,
            price,
            rating,
            description,
            content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        successful_rows = 0
        failed_rows = 0

        for i in range(0, total_rows, self.batch_size):

            batch_df = df.iloc[i : i + self.batch_size]

            for _, row in batch_df.iterrows():

                try:
                    value = (
                        str(row["pid"]),
                        str(row["product_name"]).strip(),
                        row.get("brand"),
                        row.get("main_category"),
                        row.get("sub_category"),
                        float(row.get("price")),
                        row.get("overall_rating"),
                        row.get("description"),
                        row.get("content_hash"),
                    )

                    self.db.execute_many(insert_query, [value])
                    successful_rows += 1

                except Exception as e:
                    failed_rows += 1

                    log.error(
                        "Row insert failed",
                        row_data=str(row.to_dict())[:300],
                        error=str(e),
                    )

        log.info(
            "Insertion summary",
            successful_rows=successful_rows,
            failed_rows=failed_rows,
        )

        if failed_rows > 0:
            log.warning(
                "Partial ingestion completed",
                failed_rows=failed_rows
            )