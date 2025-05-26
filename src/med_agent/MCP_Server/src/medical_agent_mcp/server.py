import asyncio
import json
from typing import Any, Dict, List, Optional

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl, BaseModel
import mcp.server.stdio

import httpx
from urllib.parse import quote_plus

# PubMed API constants
PUBMED_API_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DRUGBANK_API_BASE = "https://api.drugbank.com/v1"

# Store notes as a simple key-value dict to demonstrate state management
notes: dict[str, str] = {}

# Initialize the httpx client
http_client = httpx.AsyncClient()

server = Server("medical-agent-mcp")

async def search_pubmed(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search PubMed for articles matching the query."""
    search_url = f"{PUBMED_API_BASE}/esearch.fcgi"
    summary_url = f"{PUBMED_API_BASE}/esummary.fcgi"
    
    # First, search for article IDs
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
        "sort": "relevance"
    }
    
    async with http_client.get(search_url, params=params) as response:
        search_data = response.json()
        pmids = search_data["esearchresult"]["idlist"]
    
    if not pmids:
        return []
    
    # Then, get summaries for those articles
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json"
    }
    
    async with http_client.get(summary_url, params=params) as response:
        summary_data = response.json()
        results = []
        
        for pmid in pmids:
            article = summary_data["result"][pmid]
            results.append({
                "title": article["title"],
                "authors": ", ".join(author["name"] for author in article.get("authors", [])),
                "journal": article.get("fulljournalname", ""),
                "year": article.get("pubdate", "").split()[0],
                "abstract": article.get("abstract", "No abstract available"),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
            
        return results

async def lookup_drug(name: str) -> Dict[str, str]:
    """Mock function for drug information lookup (replace with actual API when available)"""
    # This is a mock implementation - in a real system, you would integrate with DrugBank or similar
    drug_info = {
        "name": name,
        "description": f"Information about {name}",
        "indications": "Please integrate with a proper drug database API for accurate information.",
        "contraindications": "This is mock data for demonstration purposes.",
        "side_effects": "Actual drug information should be obtained from authorized sources."
    }
    return drug_info

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available note resources."""
    return [
        types.Resource(
            uri=AnyUrl(f"note://internal/{name}"),
            name=f"Note: {name}",
            description=f"A simple note named {name}",
            mimeType="text/plain",
        )
        for name in notes
    ]

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """Read a specific note's content by its URI."""
    if uri.scheme != "note":
        raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

    name = uri.path
    if name is not None:
        name = name.lstrip("/")
        return notes[name]
    raise ValueError(f"Note not found: {name}")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """List available prompts."""
    return [
        types.Prompt(
            name="summarize-notes",
            description="Creates a summary of all notes",
            arguments=[
                types.PromptArgument(
                    name="style",
                    description="Style of the summary (brief/detailed)",
                    required=False,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """Generate a prompt by combining arguments with server state."""
    if name != "summarize-notes":
        raise ValueError(f"Unknown prompt: {name}")

    style = (arguments or {}).get("style", "brief")
    detail_prompt = " Give extensive details." if style == "detailed" else ""

    return types.GetPromptResult(
        description="Summarize the current notes",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=f"Here are the current notes to summarize:{detail_prompt}\n\n"
                    + "\n".join(
                        f"- {name}: {content}"
                        for name, content in notes.items()
                    ),
                ),
            )
        ],
    )

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools."""
    return [
        types.Tool(
            name="add-note",
            description="Add a new note",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
        ),
        types.Tool(
            name="search-pubmed",
            description="Search PubMed for medical articles",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="lookup-drug",
            description="Look up information about a medication",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution requests."""
    if not arguments:
        raise ValueError("Missing arguments")

    if name == "add-note":
        note_name = arguments.get("name")
        content = arguments.get("content")

        if not note_name or not content:
            raise ValueError("Missing name or content")

        notes[note_name] = content
        await server.request_context.session.send_resource_list_changed()

        return [
            types.TextContent(
                type="text",
                text=f"Added note '{note_name}' with content: {content}",
            )
        ]
        
    elif name == "search-pubmed":
        query = arguments.get("query")
        if not query:
            raise ValueError("Missing query")

        max_results = arguments.get("max_results", 5)
        results = await search_pubmed(query, max_results)
        
        if not results:
            return [types.TextContent(type="text", text="No results found")]
            
        formatted_results = []
        for article in results:
            formatted_results.append(
                types.TextContent(
                    type="text",
                    text=f"Title: {article['title']}\n"
                         f"Authors: {article['authors']}\n"
                         f"Journal: {article['journal']} ({article['year']})\n"
                         f"Abstract: {article['abstract']}\n"
                         f"URL: {article['url']}\n"
                )
            )
        return formatted_results
        
    elif name == "lookup-drug":
        drug_name = arguments.get("name")
        if not drug_name:
            raise ValueError("Missing drug name")
            
        drug_info = await lookup_drug(drug_name)
        return [
            types.TextContent(
                type="text",
                text=f"Drug Information for {drug_info['name']}:\n\n"
                     f"Description: {drug_info['description']}\n"
                     f"Indications: {drug_info['indications']}\n"
                     f"Contraindications: {drug_info['contraindications']}\n"
                     f"Side Effects: {drug_info['side_effects']}"
            )
        ]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="medical-agent-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())