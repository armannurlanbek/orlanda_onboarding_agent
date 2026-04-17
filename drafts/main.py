from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from agent_functions import get_tools

load_dotenv()

class Response(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    knowledge_used: list[str]
    

llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=Response)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful assistant that helps employees in Orlanda Engineering to get through onboarding process.
            Wrap the output in \n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = get_tools()
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("Enter your query: ")
raw_response = agent_executor.invoke({
    "query": query,
    "chat_history": [],
})
print(raw_response)

structured_response = parser.parse(raw_response["output"])
print(structured_response)





