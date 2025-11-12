import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore

from product_assistant.utils.model_loader import ModelLoader
from product_assistant.utils.config_loader import load_config


class DataIngestion:
    """Pipeline to load, transform, and store product review data in AstraDB."""

    def __init__(self):
        print("🔧 Initializing Data Ingestion Pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config = load_config()

    def _load_env_variables(self):
        """Load and validate required environment variables."""
        load_dotenv()

        required_vars = [
            "GROQ_API_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_DEFAULT_REGION",
            "ASTRA_DB_API_ENDPOINT",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_KEYSPACE"
        ]

        missing_vars = [v for v in required_vars if not os.getenv(v)]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        # Store them as instance variables
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_default_region = os.getenv("AWS_DEFAULT_REGION")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def _get_csv_path(self) -> str:
        """Resolve and validate the path to the product CSV."""
        csv_path = os.path.join(os.getcwd(), "data", "product_reviews.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        return csv_path

    def _load_csv(self) -> pd.DataFrame:
        """Load product CSV and validate expected columns."""
        df = pd.read_csv(self.csv_path)
        expected_columns = {
            "product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"
        }

        missing_columns = expected_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")

        return df

    def transform_data(self) -> List[Document]:
        """Transform product data into LangChain Document objects."""
        documents = []
        for _, row in self.product_data.iterrows():
            metadata = {
                "product_id": row["product_id"],
                "product_title": row["product_title"],
                "rating": row["rating"],
                "total_reviews": row["total_reviews"],
                "price": row["price"]
            }
            documents.append(Document(page_content=row["top_reviews"], metadata=metadata))

        print(f"Transformed {len(documents)} product reviews into documents.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """Store transformed documents in AstraDB vector store."""
        collection_name = self.config["astra_db"]["collection_name"]

        vstore = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        inserted_ids = vstore.add_documents(documents)
        print(f"Inserted {len(inserted_ids)} documents into AstraDB collection: '{collection_name}'.")
        return vstore, inserted_ids

    def run_pipeline(self):
        """Run the full ingestion → transformation → vector storage pipeline."""
        print("Running full ingestion pipeline...")
        documents = self.transform_data()
        vstore, _ = self.store_in_vector_db(documents)
        print("Pipeline execution completed successfully.")
