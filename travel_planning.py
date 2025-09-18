from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import ChromaDBVectorMemory
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import asyncio
import os
import time
import json
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
    SentenceTransformerEmbeddingFunctionConfig,
)
import tempfile

# Cargar variables de entorno
load_dotenv()

# Inicializamos el cliente del modelo (OpenRouter)
model_client = OpenAIChatCompletionClient(
    model=os.getenv("MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("BASE_URL"),
)

# ------------------------------
# MEMORIA VECTORIAL (ChromaDB + RAG)
# ------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="travel_memory",
            persistence_path=tmpdir,
            k=3,  # top k resultados
            score_threshold=0.4,  # similitud mínima
            embedding_function_config=SentenceTransformerEmbeddingFunctionConfig(
                model_name="all-MiniLM-L6-v2"
            ),
        )
    )


# ------------------------------
# MEMORY LAYER
# ------------------------------
async def initialize_memory() -> None:
    """Agrega información inicial a la memoria."""
    await memory.add(
        MemoryContent(
            content="Nepal tiene muchos sitios culturales, incluyendo Katmandú y Pokhara.",
            mime_type=MemoryMimeType.TEXT,
        ),
    )
    await memory.add(
        MemoryContent(
            content="Nepal ofrece oportunidades de trekking en las regiones del Annapurna y Everest.",
            mime_type=MemoryMimeType.TEXT,
        ),
    )
    await memory.add(
        MemoryContent(
            content="La gastronomía local incluye momo, dal bhat y sopa de gundruk.",
            mime_type=MemoryMimeType.TEXT,
        ),
    )


# ------------------------------
# GUARDRAILS
# ------------------------------
def guardrail_validate(response: str) -> str:
    """Valida la respuesta del modelo y asegura formato JSON básico."""
    if "badword" in response.lower():
        return "La respuesta contiene contenido inapropiado."
    try:
        json.loads(response)
    except:
        return '{"error": "La respuesta no tiene un formato JSON válido."}'
    return response


# ------------------------------
# MÉTRICAS
# ------------------------------
async def measure_agent_response(agent, task: str):
    start = time.time()
    response = await agent.run(task)
    elapsed = time.time() - start
    print(f"Agent {agent.name} took {elapsed:.2f}s to respond.")
    return response


# ------------------------------
# AGENTES (Chain of Thought)
# ------------------------------
planner_agent = AssistantAgent(
    name="planner_agent",
    model_client=model_client,
    memory=[memory],  # ← Aquí la memoria ya se usa directamente
    description="Un asistente que ayuda a planear viajes.",
    system_message="Eres un asistente útil que puede sugerir un plan de viaje basado en la solicitud del usuario.",
)

local_agent = AssistantAgent(
    name="local_agent",
    model_client=model_client,
    memory=[memory],
    description="Un asistente local que sugiere actividades y lugares interesantes para visitar.",
    system_message="Eres un asistente útil que puede sugerir actividades auténticas y lugares interesantes considerando cualquier contexto disponible.",
)

language_agent = AssistantAgent(
    name="language_agent",
    model_client=model_client,
    memory=[memory],
    description="Un asistente que proporciona consejos de idioma para el destino.",
    system_message="Eres un asistente útil que puede revisar planes de viaje y dar retroalimentación sobre cómo manejar desafíos de idioma o comunicación.",
)

travel_summary_agent = AssistantAgent(
    name="travel_summary_agent",
    model_client=model_client,
    memory=[memory],
    description="Un asistente que resume el plan de viaje final.",
    system_message="Eres un asistente útil que integra todas las sugerencias y proporciona un plan final detallado. TU RESPUESTA FINAL DEBE SER EL PLAN COMPLETO. Cuando el plan esté completo, responde con TERMINATE.",
)

termination = TextMentionTermination("TERMINATE")

group_chat = RoundRobinGroupChat(
    [
        planner_agent,
        local_agent,
        language_agent,
        travel_summary_agent,
    ],
    termination_condition=termination,
)


async def main() -> None:
    # Inicializamos memoria
    await initialize_memory()

    task = "Planea un viaje de 3 días a Nepal."

    # Medimos y ejecutamos la conversación distribuida
    raw_response = await Console(
        group_chat.run_stream(task=task),
    )
    # final_response = guardrail_validate(raw_response)

    print("\n==============================")
    print("PLAN FINAL:")
    # print(raw_response)
    for (
        msg
    ) in raw_response.messages:  # final_response es el objeto retornado por Console()
        if hasattr(msg, "content"):
            print(f"{msg.source}: {msg.content}\n")
            print("--------------------------------\n")
    print("==============================\n")

    await model_client.close()


asyncio.run(main())
