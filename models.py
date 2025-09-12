from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage
from dotenv import load_dotenv
import asyncio
import os


load_dotenv()

model_client = OpenAIChatCompletionClient(
    model=os.getenv("MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)


# Run the agent and stream the messages to the console.
async def main() -> None:
    result = await model_client.create(
        [
            UserMessage(
                content="What is the capital of France?",
                source="user",
            )
        ]
    )
    print(result)
    await model_client.close()


asyncio.run(main())
