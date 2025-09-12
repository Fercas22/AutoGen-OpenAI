import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()


def get_pizza_menu():
    return {
        "Pizzas": [
            {
                "nombre": "Margarita",
                "ingredientes": ["Tomate", "Mozzarella", "Albahaca"],
                "tamaños": ["Chica", "Mediana", "Grande"],
            },
            {
                "nombre": "Pepperoni",
                "ingredientes": ["Tomate", "Mozzarella", "Pepperoni"],
                "tamaños": ["Chica", "Mediana", "Grande"],
            },
            {
                "nombre": "Hawaiana",
                "ingredientes": ["Tomate", "Mozzarella", "Jamón", "Piña"],
                "tamaños": ["Chica", "Mediana", "Grande"],
            },
            {
                "nombre": "Cuatro Quesos",
                "ingredientes": ["Mozzarella", "Parmesano", "Gorgonzola", "Ricotta"],
                "tamaños": ["Chica", "Mediana", "Grande"],
            },
        ]
    }


async def main() -> None:
    print(os.getenv("OPENROUTER_API_KEY"))
    # Define an agent
    weather_agent = AssistantAgent(
        name="pizzeria_agent",
        model_client=OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        ),
        system_message="Eres una pizzería virtual. Solo debes responder sobre el menú de pizzas y los tamaños disponibles (Chica, Mediana, Grande). No contestes nada que no esté relacionado con pizzas.",
        tools=[get_pizza_menu],
    )

    # Define a team with a single agent and maximum auto-gen turns of 1.
    agent_team = RoundRobinGroupChat([weather_agent], max_turns=1)

    while True:
        # Get user input from the console.
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == "exit":
            break
        # Run the team and stream messages to the console.
        stream = agent_team.run_stream(task=user_input)
        await Console(stream)


asyncio.run(main())
