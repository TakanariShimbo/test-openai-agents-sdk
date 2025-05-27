import os
import asyncio
from openai.types.responses import ResponseTextDeltaEvent

from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, set_default_openai_key


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
    return f"The weather in {city} is sunny. The temperature is 20 degrees Celsius. The humidity is 50%."

# sample agent
agent = Agent(
    name="Hello world",
    instructions="You are a helpful agent.",
    tools=[get_weather],
)

# main
prompt = "What's the weather in Tokyo?"

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