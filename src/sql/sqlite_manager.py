import sqlite3
from typing import List, Dict, Any, Optional

from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException


class SQLiteManager:
    """
    SQLite manager for local database.

    Features:
    - CRUD support
    - Duplicate-safe via content_hash
    - Batch inserts
    """

    def __init__(self, db_path: str = "ecommerce.db"):
        try:
            self.db_path = db_path
            self._ensure_tables()

            log.info(
                "SQLiteManager initialized",
                db_path=self.db_path,
            )

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing SQLiteManager", e
            )

    # =====================================================
    # EXECUTE QUERY (READ / UPDATE / DELETE)
    # =====================================================

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch: bool = True,
    ) -> List[Dict[str, Any]]:

        conn = None

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            log.info(
                "Executing SQLite query",
                query=query,
                params=params,
            )

            cursor.execute(query, params or ())

            if fetch:
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

            conn.commit()
            return []

        except Exception as e:
            if conn:
                conn.rollback()

            log.error(
                "SQLite query failed",
                query=query,
                exc_info=True,
            )

            raise ProductAssistantException(
                "SQLite query failed", e
            )

        finally:
            if conn:
                conn.close()

    # =====================================================
    # EXECUTE MANY (BATCH INSERT)
    # =====================================================

    def execute_many(
        self,
        query: str,
        values: List[tuple],
    ):
        conn = None

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.executemany(query, values)
            conn.commit()

        except Exception as e:
            if conn:
                conn.rollback()

            log.error(
                "SQLite batch insert failed",
                exc_info=True,
            )

            raise ProductAssistantException(
                "SQLite batch insert failed", e
            )

        finally:
            if conn:
                conn.close()

    # =====================================================
    # TABLE CREATION
    # =====================================================

    def _ensure_tables(self):

        query = """
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            product_name TEXT NOT NULL,
            brand TEXT,
            category TEXT,
            sub_category TEXT,
            price REAL,
            rating REAL,
            description TEXT,
            content_hash TEXT UNIQUE
        );
        """

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        conn.close()

    # =====================================================
    # CREATE
    # =====================================================

    def insert_product(self, data: Dict[str, Any]):
        query = """
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
        self.execute_many(query, [(
            data["product_id"],
            data["product_name"],
            data.get("brand"),
            data.get("category"),
            data.get("sub_category"),
            data.get("price"),
            data.get("rating"),
            data.get("description"),
            data.get("content_hash"),
        )])

    # =====================================================
    # READ
    # =====================================================

    def get_product(self, product_id: str):
        return self.execute_query(
            "SELECT * FROM products WHERE product_id = ?",
            (product_id,),
        )
    def get_all_products(self, limit: int = 50):
        return self.execute_query(
            "SELECT * FROM products LIMIT ?",
            (limit,),
        )

    # =====================================================
    # UPDATE
    # =====================================================

    def update_product_price(self, product_id: str, price: float):
        query = """
        UPDATE products
        SET price = ?
        WHERE product_id = ?
        """
        self.execute_query(
            query,
            (price, product_id),
            fetch=False,
        )

    # =====================================================
    # DELETE
    # =====================================================

    def delete_product(self, product_id: str):
        query = "DELETE FROM products WHERE product_id = ?"
        self.execute_query(
            query,
            (product_id,),
            fetch=False,
        )

    def delete_all_products(self):
        self.execute_query(
            "DELETE FROM products",
            fetch=False,
        )