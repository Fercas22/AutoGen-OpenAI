import tempfile
import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Use a temporary directory for ChromaDB persistence
with tempfile.TemporaryDirectory() as tmpdir:
    chroma_user_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="preferences",
            persistence_path=tmpdir,  # Use the temp directory here
            k=2,  # Return top k results
            score_threshold=0.4,  # Minimum similarity score
            embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                model_name="all-MiniLM-L6-v2"  # Use default model for testing
            ),
        )
    )

    # Add user preferences to memory
    async def initialize_memory() -> None:
        await chroma_user_memory.add(
            MemoryContent(
                content="The weather should be in metric units",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "units"},
            )
        )

        await chroma_user_memory.add(
            MemoryContent(
                content="Meal recipe must be vegan",
                mime_type=MemoryMimeType.TEXT,
                metadata={"category": "preferences", "type": "dietary"},
            )
        )

    async def get_weather(city: str, units: str = "imperial") -> str:
        if units == "imperial":
            return f"The weather in {city} is 73 °F and Sunny."
        elif units == "metric":
            return f"The weather in {city} is 23 °C and Sunny."
        else:
            return f"Sorry, I don't know the weather in {city}."

    load_dotenv()

    model_client = OpenAIChatCompletionClient(
        model=os.getenv("MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    # Create assistant agent with ChromaDB memory
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        tools=[get_weather],
        memory=[chroma_user_memory],
    )

    # Run the agent and stream the messages to the console.
    async def main() -> None:
        await initialize_memory()
        stream = assistant_agent.run_stream(task="What is the weather in New York?")
        await Console(stream)

        await model_client.close()
        await chroma_user_memory.close()

    asyncio.run(main())
