from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

model_client = OpenAIChatCompletionClient(
    model=os.getenv("MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# Initialize user memory
user_memory = ListMemory()


# Add user preferences to memory
async def initialize_memory() -> None:
    await user_memory.add(
        MemoryContent(
            content="The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    await user_memory.add(
        MemoryContent(
            content="Meal recipe must be vegan",
            mime_type=MemoryMimeType.TEXT,
        )
    )


async def get_weather(city: str, units: str = "imperial") -> str:
    if units == "imperial":
        return f"The weather in {city} is 73 °F and Sunny."
    elif units == "metric":
        return f"The weather in {city} is 23 °C and Sunny."
    else:
        return f"Sorry, I don't know the weather in {city}."


assistant_agent = AssistantAgent(
    name="assistant_agent",
    model_client=model_client,
    tools=[get_weather],
    memory=[user_memory],
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    await initialize_memory()
    await assistant_agent._model_context.get_messages()
    stream = assistant_agent.run_stream(task="Write brief meal recipe with broth")
    await Console(stream)
    await model_client.close()


asyncio.run(main())
