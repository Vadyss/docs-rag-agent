from langchain_core.utils.uuid import uuid7
from deepagents.backends import StateBackend
from langchain.tools import tool
from embeding import vector_store
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from promts import RAG_WORKFLOW_INSTRUCTIONS, CHUNK_ANALYST_INSTRUCTIONS, SUBAGENT_DELEGATION_INSTRUCTIONS
from langchain.messages import HumanMessage

backend = StateBackend()

# Tool for searching vector database, will return 4 most relevant files
@tool(parse_docstring=True)
def search_documentation(query: str) -> str:
    """Search LangChain documentation and save matching chunks to the agent filesystem.

    Args:
        query: Natural language search query.

    Returns:
        File paths where retrieved chunks were saved under /retrieved/.
    """
    retrieved_docs = vector_store.similarity_search(query, k=4)
    batch_id = uuid7().hex[:8]
    uploads: list[tuple[str, bytes]] = []
    saved_paths: list[str] = []

    for index, doc in enumerate(retrieved_docs, start=1):
        path = f"/retrieved/{batch_id}/chunk_{index}.md"
        content = (
            f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
            f"{doc.page_content}"
        )
        uploads.append((path, content.encode("utf-8")))
        saved_paths.append(path)

    backend.upload_files(uploads)
    return (
        f"Saved {len(saved_paths)} documentation chunks:\n"
        + "\n".join(saved_paths)
    )

# Agent

max_concurrent_analysts = 3

INSTRUCTIONS = (
    RAG_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + "=" * 80
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
        max_concurrent_analysts=max_concurrent_analysts,
    )
)

chunk_analyst_subagent = {
    "name": "chunk-analyst",
    "description": (
        "Analyze one retrieved documentation chunk file. "
        "Pass the user question and a single file path under /retrieved/."
    ),
    "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
}

model = init_chat_model(model="openai:gpt-5.5")

agent = create_deep_agent(
    model=model,
    tools=[search_documentation],
    backend=backend,
    system_prompt=INSTRUCTIONS,
    subagents=[chunk_analyst_subagent],
)

EXAMPLE_QUERY = "How do I stream intermediate tool results from a subagent?"

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [HumanMessage(content=EXAMPLE_QUERY)]}
    )

    for msg in result.get("messages", []):
        if msg.text:
            print(msg.text)