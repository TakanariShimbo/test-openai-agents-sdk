import os
import asyncio

from dotenv import load_dotenv
from agents import Agent, Runner, WebSearchTool, set_default_openai_key
from openai.types.responses import ResponseTextDeltaEvent


# load .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# set openai key
set_default_openai_key(api_key)

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),
    ],
    model="gpt-4o",
)
# main
prompt = "最近のユニオンツールのニュースを教えてください。"

async def main():
    result = await Runner.run(agent, input=prompt)
    print(result.final_output)

async def main_stream():
    result = Runner.run_streamed(agent, input=prompt)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main_stream())