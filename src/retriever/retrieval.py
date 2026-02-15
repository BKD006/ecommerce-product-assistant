import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter

from product_assistant.utils.config_loader import load_config
from product_assistant.utils.model_loader import ModelLoader


class Retriever:
    """
    Retriever pipeline for querying AstraDB vector store with contextual compression.
    
    This class:
      - Loads environment variables for AstraDB access.
      - Initializes embeddings and vector store.
      - Creates an MMR-based retriever.
      - Applies an LLM-based compression filter to refine retrieved context.
    """

    def __init__(self):
        """Initialize retriever with config and environment setup."""
        print("🔍 Initializing Retriever Pipeline...")
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever_instance = None

    def _load_env_variables(self):
        """
        Load and validate environment variables required for AstraDB and AWS.
        Ensures that all critical credentials are present before initialization.
        """
        load_dotenv()

        required_vars = [
            "AWS_SECRET_ACCESS_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_DEFAULT_REGION",
            "ASTRA_DB_API_ENDPOINT",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_KEYSPACE",
        ]

        missing_vars = [v for v in required_vars if not os.getenv(v)]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        # Assign to instance variables
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_default_region = os.getenv("AWS_DEFAULT_REGION")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def load_retriever(self):
        """
        Initialize the vector store retriever with MMR search and LLM-based compression.
        
        Steps:
          1. Connect to AstraDB Vector Store.
          2. Create an MMR retriever for diverse search results.
          3. Load an LLM-based compressor to refine retrieved chunks.
        """
        # Step 1: Initialize vector store (if not already loaded)
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]

            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
            )
            print(f"Connected to AstraDB collection: '{collection_name}'")

        # Step 2: Build retriever with MMR search
        if not self.retriever_instance:
            retriever_config = self.config.get("retriever", {})
            top_k = retriever_config.get("top_k", 3)

            mmr_retriever = self.vstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,          # Number of final retrieved results
                    "fetch_k": 20,       # Number of candidates fetched before MMR pruning
                    "lambda_mult": 0.7,  # Diversity factor
                    "score_threshold": 0.6,  # Minimum similarity threshold
                },
            )
            print("🔧 MMR retriever initialized successfully.")

            # Step 3: Apply contextual compression
            llm = self.model_loader.load_llm()
            compressor = LLMChainFilter.from_llm(llm)
            self.retriever_instance = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=mmr_retriever,
            )
            print("LLM-based compression retriever configured successfully.")

        return self.retriever_instance

    def call_retriever(self, query: str):
        """
        Execute the retriever pipeline for a given query.
        
        Args:
            query (str): User query text.
        
        Returns:
            List[Document]: A list of contextually relevant LangChain Document objects.
        """
        retriever = self.load_retriever()
        print(f"Querying retriever for: '{query}'")
        output = retriever.invoke(query)
        print(f"Retrieved {len(output)} relevant documents.")
        return output
