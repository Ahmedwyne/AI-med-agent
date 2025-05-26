# Launch script for MCP server
import asyncio
from medical_agent_mcp.server import main

if __name__ == "__main__":
    asyncio.run(main())
