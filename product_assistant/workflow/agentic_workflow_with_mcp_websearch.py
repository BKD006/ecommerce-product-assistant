"""
Agentic RAG pipeline that integrates:
- LangGraph for agent workflow
- MCP (Model Context Protocol) for hybrid search and web tools
- LLM-based grading, rewriting, and answer generation

This workflow performs:
1. Query analysis
2. Conditional retrieval via MCP retriever
3. Document relevance grading
4. Result generation or query rewriting
5. Fallback web search if needed
"""

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
    """
    Autonomous RAG agent using LangGraph + MCP tools.

    Features:
    - Detects when to call retrieval tools.
    - Retrieves product information via MCP hybrid_search server.
    - Grades retrieved documents for relevance.
    - Performs generation or query rewriting.
    - Optionally performs web search via MCP tool.
    """

    class AgentState(TypedDict):
        """TypedDict defining the state passed between graph nodes."""
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self):
        """Initialize LLM, MCP client, retrievers, workflow graph, and memory."""

        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        # Configure MCP client pointing to your hybrid search server
        self.mcp_client = MultiServerMCPClient(
            {
                "hybrid_search": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/mcp",
                }
            }
        )

        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        # Asynchronously load MCP tools safely
        asyncio.run(self._safe_async_init())

    async def async_init(self):
        """Direct async initializer (unused externally)."""
        self.mcp_tools = await self.mcp_client.get_tools()

    async def _safe_async_init(self):
        """
        Safely initialize MCP tools.
        Ensures workflow does not break if MCP server is offline.
        """
        try:
            self.mcp_tools = await self.mcp_client.get_tools()
            print("MCP tools loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load MCP tools — {e}")
            self.mcp_tools = []

    # -------------------------------------------------------------------------
    # Node 1: AI Assistant Routing Logic
    # -------------------------------------------------------------------------
    def _ai_assistant(self, state: AgentState):
        """
        Decide whether to answer directly or route to retriever.

        Simple keyword-based routing:
        - If the query seems product-related → use retrieval
        - Otherwise → answer directly with LLM
        """
        print("--- CALL ASSISTANT ---")

        last_message = state["messages"][-1].content.lower()

        # Keyword-based intent detection
        if any(word in last_message for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}

        # Normal LLM-based response
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the user directly.\n\n"
            "Question: {question}\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({"question": last_message}) or "I'm not sure about that."

        return {"messages": [HumanMessage(content=response)]}

    # -------------------------------------------------------------------------
    # Node 2: Vector Retriever via MCP
    # -------------------------------------------------------------------------
    async def _vector_retriever(self, state: AgentState):
        """
        Retrieve product information using MCP tool `get_product_info`.

        Args:
            state: Agent state including user query.

        Returns:
            Retrieved product context embedded in a HumanMessage.
        """
        print("--- RETRIEVER (MCP) ---")

        query = state["messages"][-1].content

        # Find MCP retriever tool
        tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if not tool:
            return {"messages": [HumanMessage(content="Retriever tool not found in MCP client.")]}

        try:
            result = await tool.ainvoke({"query": query})
            context = result or "No relevant product data found."
        except Exception as e:
            context = f"Error invoking retriever: {e}"

        return {"messages": [HumanMessage(content=context)]}

    # -------------------------------------------------------------------------
    # Node 3: Web Search via MCP (Fallback)
    # -------------------------------------------------------------------------
    async def _web_search(self, state: AgentState):
        """
        Perform web search using MCP tool `web_search`.

        Args:
            state: Current workflow state.

        Returns:
            Web search result message.
        """
        print("--- WEB SEARCH (MCP) ---")

        query = state["messages"][-1].content

        # Identify tool
        tool = next(t for t in self.mcp_tools if t.name == "web_search")

        result = await tool.ainvoke({"query": query})
        context = result or "No data from web"

        return {"messages": [HumanMessage(content=context)]}

    # -------------------------------------------------------------------------
    # Node 4: Grade Retrieved Documents
    # -------------------------------------------------------------------------
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        """
        Grade retrieved content for relevance.

        If relevant → proceed to answer generation.
        If irrelevant → rewrite the query for better retrieval.

        Returns:
            "generator" | "rewriter"
        """
        print("--- GRADER ---")

        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template=(
                "You are a grader.\n\n"
                "Question: {question}\nDocs: {docs}\n\n"
                "Are docs relevant to the question? Answer yes or no."
            ),
            input_variables=["question", "docs"],
        )

        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs}) or ""

        return "generator" if "yes" in score.lower() else "rewriter"

    # -------------------------------------------------------------------------
    # Node 5: Generate Final Answer
    # -------------------------------------------------------------------------
    def _generate(self, state: AgentState):
        """
        Generate final enriched answer using PRODUCT_BOT prompt.

        Args:
            state: Messages including question + context.

        Returns:
            HumanMessage with generated response.
        """
        print("--- GENERATE ---")

        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({"context": docs, "question": question}) or "No response generated."
        except Exception as e:
            response = f"Error generating response: {e}"

        return {"messages": [HumanMessage(content=response)]}

    # -------------------------------------------------------------------------
    # Node 6: Rewrite Query
    # -------------------------------------------------------------------------
    def _rewrite(self, state: AgentState):
        """
        Rewrite the user's query to improve search retrieval.

        Args:
            state: Agent state containing original query.

        Returns:
            HumanMessage with rewritten query.
        """
        print("--- REWRITE ---")

        question = state["messages"][0].content

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this user query to make it more clear and specific for a search engine.\n"
            "Do NOT answer the query. Only rewrite it.\n\n"
            "Query: {question}\nRewritten Query:"
        )

        chain = prompt | self.llm | StrOutputParser()

        try:
            rewritten = chain.invoke({"question": question}).strip()
        except Exception as e:
            rewritten = f"Error rewriting query: {e}"

        return {"messages": [HumanMessage(content=rewritten)]}

    # -------------------------------------------------------------------------
    # Workflow Builder
    # -------------------------------------------------------------------------
    def _build_workflow(self):
        """
        Construct LangGraph workflow with nodes and conditional edges.

        Returns:
            StateGraph: Completed workflow graph.
        """
        workflow = StateGraph(self.AgentState)

        # Define nodes
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("WebSearch", self._web_search)

        # START → Assistant
        workflow.add_edge(START, "Assistant")

        # Assistant → Retriever OR END
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )

        # Retriever → Generator OR Rewriter
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )

        # Rewriter → WebSearch → Generator
        workflow.add_edge("Rewriter", "WebSearch")
        workflow.add_edge("WebSearch", "Generator")

        # Final step
        workflow.add_edge("Generator", END)

        return workflow

    # -------------------------------------------------------------------------
    # Execution API
    # -------------------------------------------------------------------------
    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        """
        Execute the agentic RAG workflow asynchronously.

        Args:
            query (str): User input query.
            thread_id (str): ID for memory thread tracking.

        Returns:
            str: Final generated response.
        """
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content
