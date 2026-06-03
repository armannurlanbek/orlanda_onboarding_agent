"""
Run the RAG agent with a test query.
"""
from rag_agent.agent import get_base_agent
from rag_agent.agent import Context

config = {"configurable": {"thread_id": "1"}}


def main():
    # Example query – agent will use retrieve_context if it needs knowledge base
    response = get_base_agent().invoke(
        {
            "messages": [
                {"role": "user", "content": "What information do you have in the knowledge base? Summarize any key topics."}
            ]
        },
        config=config,
        context=Context(user_id="1"),
    )

    print(response["structured_response"])


if __name__ == "__main__":
    main()
