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


class AgenticRAG:
    """
    An agentic Retrieval-Augmented Generation (RAG) agent using LangGraph.

    This pipeline:
      - Accepts a user query.
      - Determines whether to use a retriever based on the query content.
      - Retrieves relevant documents and evaluates their relevance.
      - Generates a final answer using retrieved context or rewrites the query and retries.
    
    Workflow Nodes:
        - Assistant (AI Core Agent)
        - Retriever (Document Retrieval)
        - Generator (RAG Response Generation)
        - Rewriter (Reformulate Unhelpful Queries)
    """

    class AgentState(TypedDict):
        """TypedDict for graph state management."""
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        """Initialize components for RAG."""
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _format_docs(self, docs) -> str:
        """
        Format retrieved documents into readable chunks.

        Args:
            docs (list): List of LangChain Document objects.

        Returns:
            str: Human-readable string representation of document chunks.
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

    def _ai_assistant(self, state: AgentState):
        """
        Decide whether to invoke retriever or respond directly.

        Args:
            state (AgentState): Current chat state.
        
        Returns:
            dict: Next message or tool request for retriever.
        """
        print("--- CALL ASSISTANT ---")
        last_message = state["messages"][-1].content.lower()

        # Trigger retriever if query pertains to products or reviews
        if any(word in last_message for word in ["price", "review", "product"]):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        
        # Default direct response using LLM
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"question": last_message})
        return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        """
        Retrieve relevant documents using the vector store retriever.

        Args:
            state (AgentState): Current chat state.
        
        Returns:
            dict: Formatted document context as next message.
        """
        print("--- RETRIEVER ---")
        query = state["messages"][-1].content

        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)

        formatted_docs = self._format_docs(docs)
        return {"messages": [HumanMessage(content=formatted_docs)]}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        """
        Evaluate whether the fetched documents are relevant to the query.

        Args:
            state (AgentState): Current chat state.

        Returns:
            Literal["generator", "rewriter"]: Next node to trigger.
        """
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})

        return "generator" if "yes" in score.lower() else "rewriter"

    def _generate(self, state: AgentState):
        """
        Generate a response based on query and retrieved context.

        Args:
            state (AgentState): Current chat state.

        Returns:
            dict: Generated answer message.
        """
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        # Use a predefined product assistant prompt from prompt registry
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})

        return {"messages": [HumanMessage(content=response)]}

    def _rewrite(self, state: AgentState):
        """
        Rewrite unclear or unproductive queries before retrying.

        Args:
            state (AgentState): Current chat state.

        Returns:
            dict: Message containing the rewritten query.
        """
        print("--- REWRITE ---")
        question = state["messages"][0].content

        new_query = self.llm.invoke(
            [HumanMessage(content=f"Rewrite the query to be clearer: {question}")]
        )
        return {"messages": [HumanMessage(content=new_query.content)]}

    def _build_workflow(self):
        """
        Construct the LangGraph-based RAG workflow.
        
        Returns:
            StateGraph: Configured processing workflow.
        """
        workflow = StateGraph(self.AgentState)

        # Define pipeline nodes
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        # Define state transitions
        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")

        return workflow

    def run(self, query: str, thread_id: str = "default_thread") -> str:
        """
        Run the RAG agent on a given query.

        Args:
            query (str): User query string.
            thread_id (str): Optional conversational thread ID for memory.

        Returns:
            str: Final answer generated by the agent.
        """
        result = self.app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        return result["messages"][-1].content
