from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from datetime import datetime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv
from rag_agent import vector_store

load_dotenv()


model = init_chat_model(
    model = "gpt-4o-mini",
    temperature = 0.5,
    timeout = 10,
    max_tokens = 1000
)


system_prompt = """
You are a helpful assistant that helps employees in Orlanda Engineering to get through onboarding process. You have access to tools:
- get_current_time: use to get the current time 
- retrieve_context: use to retrieve context from the knowledge base to help answer the query.
"""

@tool(description="Use this tool to get the current time .")
def get_current_time(runtime: ToolRuntime) -> str:
    """
    Use this tool to get the current time 

    """
    return datetime.now().isoformat()

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """
    Retrieve context from the knowledge base to help answer the query.
    """
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source:{doc.metadata}\nContent:{doc.page_content}")
        for doc in retrieved_docs
    )

    return (serialized, serialized)

@dataclass
class Context:
    """Custom runtime context schema"""
    user_id: str

@dataclass
class ResponseFormat:
    """Response format for the agent"""
    response_content: str


agent = create_agent(
    model = model,
    tools = [get_current_time, retrieve_context],
    system_prompt = system_prompt,
    checkpointer = InMemorySaver(),
    context_schema = Context,
    response_format = ResponseFormat
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What to do to define and indicate the area for aluminum panel?"}]},
    config = config,
    context = Context(user_id = "1")
)

print(response['structured_response'])
