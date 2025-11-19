from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

class AgenticRAG:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
    
    def __init__(self):
        self.retriever= Retriever()
        self.model_loader= ModelLoader()
        self.llm= self.model_loader.load_llm()
        self.checkpointer=MemorySaver()

        self.mcp_client= MultiServerMCPClient(
            {
                "hybrid_search":{
                    "transport": "streamable_http",
                    "url": "http://localhost:8080/mcp"
                }
            }
        )
        self.workflow= self._build_workflow()
        self.app=self.workflow.compile(checkpointer=self.checkpointer)

        asyncio.run(self._safe_async_init())

    async def _safe_async_init(self):
        """Safe async init wrapper (prevents event loop crash)."""
        try:
            self.mcp_tools = await self.mcp_client.get_tools()
            print("MCP tools loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load MCP tools — {e}")
            self.mcp_tools = []

    def _ai_assistant(self, state: AgentState):
        pass

    async def _vector_retriever(self, state: AgentState):
        pass

    async def _web_search(self, state: AgentState):
        pass

    def _grade_documents(self, state: AgentState):
        pass

    def _generate(self, state: AgentState):
        pass

    def _rewrite(self, state: AgentState):
        pass

    def _build_workflow(self, state: AgentState):
        pass

    def run(self, query:str, thread_id= "default_thread")->str:
        pass