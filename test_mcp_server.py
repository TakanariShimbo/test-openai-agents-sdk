import asyncio
import os
import json
from contextlib import AsyncExitStack
from typing import Any

from agents import Agent, Runner, set_default_openai_key
from agents.mcp import MCPServerStdio
from dotenv import load_dotenv

# load .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# set openai key
set_default_openai_key(api_key)

# params
params_json_str = """
{
    "mcpServers": {
        "playwright": {
            "command": "npx", 
            "args": [
                "-y", 
                "@playwright/mcp@latest"
            ]
        },
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "C:/Users/xxx"
            ]   
        }
    }
}
"""

def parse_mcp_servers_json(json_str: str) -> dict[str, Any]:
    try:
        mcp_servers = json.loads(json_str)
        if not isinstance(mcp_servers, dict):
            raise AssertionError(f"MCP servers must be a dict.")

        mcp_servers_params = mcp_servers.get("mcpServers", {})
        if not isinstance(mcp_servers_params, dict):
            raise AssertionError(f"MCP servers must be a dict.")

        for name, params in mcp_servers_params.items():
            if not isinstance(name, str):
                raise AssertionError(f"MCP server 'name' must be a string.")
            if not isinstance(params, dict):
                raise AssertionError(f"MCP server 'params' must be a dict.")
            if params.get("command", None) is None:
                raise AssertionError(f"MCP server 'command' must be set.")
            if params.get("args", None) is None:
                raise AssertionError(f"MCP server 'args' must be set.")
    except (json.JSONDecodeError, AssertionError) as e:
        raise ValueError(str(e))
    
    return mcp_servers_params

async def main():
    params_dict = parse_mcp_servers_json(params_json_str)

    # MCP servers
    async with AsyncExitStack() as stack:
        mcp_servers: list[MCPServerStdio] = []
        for _, params in params_dict.items():
            server = await stack.enter_async_context(
                MCPServerStdio(
                    params=params,
                    client_session_timeout_seconds=30,
                )
            )
            await server.list_tools()
            mcp_servers.append(server)

        # Create agent
        agent = Agent(
            name="Assistant",
            instructions="あなたは親切なアシスタントです。",
            mcp_servers=mcp_servers,
        )

        result = await Runner.run(
            agent, 
            "今日の東京都の天気をPlaywrightで調べ、その結果を教えてください。〇〇へ保存してください。",
        )
        print(result.final_output)

    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
