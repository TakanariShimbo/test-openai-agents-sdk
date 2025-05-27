import os
from dotenv import load_dotenv
from agents import Agent, Runner, set_default_openai_key


# load .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# set openai key
set_default_openai_key(api_key)

# sample agent sdk
agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)