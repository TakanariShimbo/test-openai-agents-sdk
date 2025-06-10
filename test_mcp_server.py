import asyncio
import os

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


async def main():
    async with MCPServerStdio(
        name="Playwright MCP Server",
        params={
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest"],
        },
        client_session_timeout_seconds=30,
    ) as server:
        # await server.list_tools()

        agent = Agent(
            name="Assistant",
            instructions="あなたは親切なアシスタントです。",
            mcp_servers=[server],
        )

        result = await Runner.run(
            agent, "明日の東京の天気をPlaywrightを使って調べ、その結果を教えてください。"
        )
        print(result.final_output)

    # wait to finish the background task
    await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())