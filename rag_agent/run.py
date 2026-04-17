"""
Run the RAG agent with a test query.
"""
from rag_agent.agent import agent
from rag_agent.agent import Context

config = {"configurable": {"thread_id": "1"}}

# Example query – agent will use retrieve_context if it needs knowledge base
response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What information do you have in the knowledge base? Summarize any key topics."}
        ]
    },
    config=config,
    context=Context(user_id="1"),
)

print(response["structured_response"])
