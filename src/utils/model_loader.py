import os
import sys
from dotenv import load_dotenv
from src.utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from sentence_transformers import SentenceTransformer
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException
import asyncio

class APIKeyManager:
    def __init__(self):
        self.api_keys={
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
                }
        
        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded from environment")
            else:
                log.warning(f"{key} is missing from environment")

    def get(self, key:str):
        return self.api_keys.get(key)
    
# =====================================================
# Local BGE Embedding Wrapper
# =====================================================

# class LocalBGEEmbeddings:
#     def __init__(self, model_name: str, normalize: bool = True):
#         self.model = SentenceTransformer(model_name)
#         self.normalize = normalize
#         log.info("Loaded local embedding model", model=model_name)

#     def embed_documents(self, texts):
#         return self.model.encode(
#             texts,
#             normalize_embeddings=self.normalize
#         ).tolist()

#     def embed_query(self, text):
#         return self.model.encode(
#             text,
#             normalize_embeddings=self.normalize
#         ).tolist()

# =====================================================
# Model Loader
# =====================================================

class ModelLoader:

    def __init__(self):
        self.api_key_mgr= APIKeyManager()
        self.config= load_config()
        log.info("YAML config loaded", config_keys= list(self.config.keys()))

    def load_embeddings(self, embedding_type: str = "product"):
            """
            embedding_type:
                - "product"
                - "policy"
            """

            try:
                embedding_block = self.config["embedding_models"][embedding_type]
                provider = embedding_block["provider"]
                model_name = embedding_block["model_name"]

                log.info(
                    "Loading embedding model",
                    provider=provider,
                    model=model_name,
                    type=embedding_type,
                )

                # if provider == "local":
                #     return LocalBGEEmbeddings(
                #         model_name=model_name,
                #         normalize=embedding_block.get("normalize", True),
                #     )

                if provider == "aws":
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        asyncio.set_event_loop(asyncio.new_event_loop())

                    return BedrockEmbeddings(model_id=model_name)

                else:
                    raise ValueError(f"Unsupported embedding provider: {provider}")

            except Exception as e:
                log.error("Error loading embedding model", error=str(e))
                raise ProductAssistantException(
                    "Failed to load embedding model", sys
                )
    

    def load_llm(self):
        llm_block= self.config["llm"]
        provider_key= os.getenv("LLM_PROVIDER", "groq")
        if provider_key not in llm_block:
            log.error("LLM Provider not found in config", provider=provider_key)
            raise ValueError(f"LLM Provider '{provider_key}' not found in config")
        
        llm_config= llm_block[provider_key]
        provider= llm_config.get("provider")
        model_name=llm_config.get("model_name")
        temperature= llm_config.get("temperature", 0.2)

        log.info("Loading LLM", provider=provider_key, model= model_name)

        if provider=="groq":
            return ChatGroq(model=model_name,
                            api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                            temperature=temperature
                            )
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")
    

    def load_reranker_llm(self):
        llm_block= self.config["llm"]
        provider_key= "groq_reranker"
        if provider_key not in llm_block:
            log.error("LLM Provider not found in config", provider=provider_key)
            raise ValueError(f"LLM Provider '{provider_key}' not found in config")
        
        llm_config= llm_block[provider_key]
        provider= llm_config.get("provider")
        model_name=llm_config.get("model_name")
        temperature= llm_config.get("temperature")

        log.info("Loading LLM", provider=provider_key, model= model_name)

        if provider=="groq":
            return ChatGroq(model=model_name,
                            api_key=self.api_key_mgr.get("GROQ_API_KEY"),
                            temperature=temperature
                            )
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")