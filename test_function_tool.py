import os
import asyncio

from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_default_openai_key
from openai.types.responses import ResponseTextDeltaEvent


# load .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# set openai key
set_default_openai_key(api_key)


# sample function tool
@function_tool
def get_weather(city: str) -> str:
    return f"{city}の天気は晴れです。温度は20度です。湿度は50%です。"

# sample agent
agent = Agent(
    name="Assistant",
    instructions="あなたは親切なアシスタントです。",
    tools=[get_weather],
)

# main
prompt = "東京の天気を教えてください。"

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