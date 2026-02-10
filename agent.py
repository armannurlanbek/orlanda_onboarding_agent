from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from datetime import datetime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from dotenv import load_dotenv
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
- get_corporate_information: use to get the corporate information on Orlanda Engineering
"""

@tool(description="Use this tool to get the current time .")
def get_current_time(runtime: ToolRuntime) -> str:
    """
    Use this tool to get the current time 

    """
    return datetime.now().isoformat()

@dataclass
class Context:
    """Custom runtime context schema"""
    user_id: str

@dataclass
class ResponseFormat:
    """Response format for the agent"""
    punny_response: str
    current_time: str | None = None


agent = create_agent(
    model = model,
    tools = [get_current_time],
    system_prompt = system_prompt,
    checkpointer = InMemorySaver(),
    context_schema = Context,
    response_format = ResponseFormat
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the current time?"}]},
    config = config,
    context = Context(user_id = "1")
)

print(response['structured_response'])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Thank you"}]},
    config = config,
    context = Context(user_id = "1")
)

print(response['structured_response'])