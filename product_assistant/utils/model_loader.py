import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_aws import BedrockEmbeddings
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import ProductAssistantException
import asyncio

class APIKeyManager:
    def __init__(self):
        self.api_keys={
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_DEFAULT_REGION": os.getenv("AWS_DEFAULT_REGION"),
        "ASTRA_DB_API_ENDPOINT": os.getenv("ASTRA_DB_API_ENDPOINT"),
        "ASTRA_DB_APPLICATION_TOKEN": os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
        "ASTRA_DB_KEYSPACE": os.getenv("ASTRA_DB_KEYSPACE")
                }
        
        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded from environment")
            else:
                log.warning(f"{key} is missing from environment")

    def get(self, key:str):
        return self.api_keys.get(key)


class ModelLoader:

    def __init__(self):
        self.api_key_mgr= APIKeyManager()
        self.config= load_config()
        log.info("YAML config loaded", config_keys= list(self.config.keys()))

    def load_embeddings(self):
        try:
            model_name= self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            return BedrockEmbeddings(model_id=model_name)
        
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ProductAssistantException("Failed to load embedding model", sys)
        

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