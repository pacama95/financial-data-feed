"""MCP Server for Financial News RAG System.

Exposes tools for AI agents to search and analyze financial news.
"""

__all__ = ["create_server", "run_server"]


def create_server():
    """Create the MCP server."""
    from mcp_server.server import create_server as _create_server
    return _create_server()


def run_server():
    """Run the MCP server."""
    from mcp_server.server import run_server as _run_server
    return _run_server()
