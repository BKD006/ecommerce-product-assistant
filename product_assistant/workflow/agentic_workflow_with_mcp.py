"""
Agentic RAG system built using LangGraph, MCP tool integration, 
and LLM-driven retrieval-augmentation workflow.

This module defines an autonomous agent that:
1. Decides whether to call a retriever or answer directly.
2. Retrieves product data via MCP tools.
3. Grades retrieved documents for relevance.
4. Generates a final answer or rewrites the query.
"""

from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from product_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from product_assistant.retriever.retrieval import Retriever
from product_assistant.utils.model_loader import ModelLoader
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


class AgenticRAG:
    """
    Agentic Retrieval-Augmented Generation (RAG) pipeline using LangGraph.
    Includes:
    - Autonomous decision-making
    - Vector retrieval
    - Query rewriting
    - MCP-integrated product retrieval
    """

    class AgentState(TypedDict):
        """State object passed between LangGraph nodes."""
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        """Initialize RAG components, LLM, MCP client, and workflow."""
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        # MCP client for external tools (product retriever)
        self.mcp_client = MultiServerMCPClient({
            "product_retriever": {
                "command": "python",
                "args": ["product_assistant/mcp_servers/product_search_server.py"],
                "transport": "stdio"
            }
        })

        # Load available MCP tools
        self.mcp_tools = asyncio.run(self.mcp_client.get_tools())

        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # -------------------------------------------------------------------------
    # Helper: Document Formatting
    # -------------------------------------------------------------------------
    def _format_docs(self, docs) -> str:
        """
        Format retrieved product documents into readable text.

        Args:
            docs (list): Retrieved documents from vector store or MCP tool.

        Returns:
            str: Human-readable formatted string.
        """
        if not docs:
            return "No relevant documents found."

        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)

        return "\n\n---\n\n".join(formatted_chunks)

    # -------------------------------------------------------------------------
    # Node: Decide whether to answer directly or use retriever
    # -------------------------------------------------------------------------
    def _ai_assistant(self, state: AgentState):
        """
        Main assistant decision function. Determines whether retrieval is required.

        Rules:
        - If query contains product-related keywords (price, review, product), 
          it routes to the retriever.
        - Otherwise, answers directly using the LLM.

        Args:
            state (AgentState): Current execution state.

        Returns:
            dict: Updated messages for next node.
        """
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content.lower()

        # Detect product-related queries
        if any(word in last_message for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}

        # Otherwise respond normally
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the user directly.\n\n"
            "Question: {question}\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": messages[-1].content})

        return {"messages": [HumanMessage(content=response)]}

    # -------------------------------------------------------------------------
    # Node: Retrieve via MCP Tool
    # -------------------------------------------------------------------------
    def _vector_retriever(self, state: AgentState):
        """
        Call the MCP tool `get_product_info` to retrieve context for the query.

        Args:
            state (AgentState): Current graph state.

        Returns:
            dict: Retrieved content injected as message.
        """
        print("--- RETRIEVER (MCP) ---")
        query = state["messages"][-1].content

        # Identify MCP tool
        tool = next(t for t in self.mcp_tools if t.name == "get_product_info")

        # Execute async MCP tool call synchronously
        result = asyncio.run(tool.ainvoke({"query": query}))
        context = result or "No data"

        return {"messages": [HumanMessage(content=context)]}

    # -------------------------------------------------------------------------
    # Node: Grade Documents
    # -------------------------------------------------------------------------
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        """
        Grade whether retrieved docs are relevant to the original question.

        If relevant → generate final answer.
        If irrelevant → rewrite query to improve retrieval.

        Args:
            state (AgentState): State containing original question + docs.

        Returns:
            Literal["generator", "rewriter"]: Next node name.
        """
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template=(
                "You are a grader. Determine if the retrieved documents are relevant.\n"
                "Question: {question}\nDocs: {docs}\n\n"
                "Answer yes or no."
            ),
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})

        return "generator" if "yes" in score.lower() else "rewriter"

    # -------------------------------------------------------------------------
    # Node: Generate Final Answer
    # -------------------------------------------------------------------------
    def _generate(self, state: AgentState):
        """
        Generate a final answer using retrieved context.

        Args:
            state (AgentState): Current graph state containing context + question.

        Returns:
            dict: Final answer message.
        """
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})

        return {"messages": [HumanMessage(content=response)]}

    # -------------------------------------------------------------------------
    # Node: Rewrite Query
    # -------------------------------------------------------------------------
    def _rewrite(self, state: AgentState):
        """
        Rewrite unclear or unsearchable user queries for improved retrieval.

        Args:
            state (AgentState): Current graph state.

        Returns:
            dict: Rewritten query.
        """
        print("--- REWRITE ---")
        question = state["messages"][0].content

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this user query to be clear and specific for a search engine.\n"
            "Do NOT answer the query.\n\nQuery: {question}\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()
        rewritten = chain.invoke({"question": question})

        return {"messages": [HumanMessage(content=rewritten.strip())]}

    # -------------------------------------------------------------------------
    # Build LangGraph Workflow
    # -------------------------------------------------------------------------
    def _build_workflow(self):
        """
        Build the complete LangGraph workflow defining all agent nodes and routing.

        Returns:
            StateGraph: Configured workflow graph ready for compilation.
        """
        workflow = StateGraph(self.AgentState)

        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        # Flow: START → Assistant
        workflow.add_edge(START, "Assistant")

        # Assistant → Retriever OR END
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )

        # Retriever → Generator or Rewriter
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )

        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")

        return workflow

    # -------------------------------------------------------------------------
    # Public Execution Method
    # -------------------------------------------------------------------------
    def run(self, query: str, thread_id: str = "default_thread") -> str:
        """
        Execute the entire Agentic RAG flow.

        Args:
            query (str): User query string.
            thread_id (str): Thread identifier for memory persistence.

        Returns:
            str: Final assistant response.
        """
        result = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content
