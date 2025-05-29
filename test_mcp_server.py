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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(current_dir, "sample")

    async with MCPServerStdio(
        name="npxを使用したファイルシステムサーバー",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", sample_dir],
        },
    ) as server:
        await server.list_tools()

        agent = Agent(
            name="Assistant",
            instructions="ツールを使用してファイルシステムを読み取り、それらのファイルに基づいて質問に答えてください。",
            mcp_servers=[server],
        )

        result = await Runner.run(
            agent, "「ohtani-san.txt」ファイルの中に何と記載されている？"
        )
        print(result.final_output)


async def run_main_with_proper_cleanup():
    try:
        await main()
    finally:
        # wait to finish the background task
        await asyncio.sleep(0.1)
        
        # if there are remaining tasks, cancel them
        loop = asyncio.get_running_loop()
        pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done() and task != asyncio.current_task()]
        
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            
            # wait for the cancelled tasks to finish
            await asyncio.gather(*pending_tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(run_main_with_proper_cleanup())