import os
import base64
import asyncio

from dotenv import load_dotenv
from agents import Agent, Runner, set_default_openai_key
from openai.types.responses import ResponseTextDeltaEvent


# load .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# set openai key
set_default_openai_key(api_key)

# sample agent
agent = Agent(
    name="Assistant",
    instructions="あなたは親切なアシスタントです。",
)

# main
def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

image_path = os.path.join(os.path.dirname(__file__), "image_bison.jpg")
b64_image = image_to_base64(image_path)
prompt = [
    {
        "role": "user",
        "content": [
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:image/jpeg;base64,{b64_image}",
            }
        ],
    },
    {
        "role": "user",
        "content": "What do you see in this image?",
    },
]

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