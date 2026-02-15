from mcp.server.fastmcp import FastMCP
from retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize MCP Server
mcp = FastMCP("hybrid_search")

# Load Retriever once during server initialization
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# Initialize DuckDuckGo search tool
duckduckgo = DuckDuckGoSearchRun()

def format_docs(docs) -> str:
    """
    Helper function to format retrieved documents into readable text.

    Args:
        docs (list): List of document objects.

    Returns:
        str: Formatted string representation of documents with metadata like title, price, rating, and reviews.
    """
    if not docs:
        return ""
    
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
    
    return "\n\n----\n\n".join(formatted_chunks)

# MCP Tools (Endpoints)

@mcp.tool()
async def get_product_info(query: str) -> str:
    """
    Retrieve product information based on a query using the local vector store.

    Args:
        query (str): Search query for product retrieval.

    Returns:
        str: Formatted product details or an error message.
    """
    try:
        docs = retriever.invoke(query)
        context = format_docs(docs)
        if not context.strip():
            return "No local results found"
        return context
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"

@mcp.tool()
async def web_search(query: str) -> str:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query (str): Search query string for the external web search.

    Returns:
        str: Search result text or an error message.
    """
    try:
        return duckduckgo.run(query)
    except Exception as e:
        return f"Error during web search: {str(e)}"

# Run MCP Server
if __name__ == "__main__":
    # Run server with streamable HTTP transport for interactive requests
    mcp.run(transport="streamable-http")
