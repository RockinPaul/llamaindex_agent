from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
import chromadb
import os

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b


load_dotenv()
openai_api_key = os.getenv("OPEN_AI_API_KEY")

# initialize llm
llm = OpenAI(
    model_name="o4-mini-2025-04-16",
    temperature=0.7,
    max_tokens=100,
    api_key=openai_api_key,
)


async def main():
    # # initialize agent
    # agent = AgentWorkflow.from_tools_or_functions(
    #     [FunctionTool.from_defaults(multiply)], llm=llm
    # )

    # # stateless
    # response = await agent.run("What is 2 times 2?")
    # print(response)

    # # remembering state
    # from llama_index.core.workflow import Context

    # ctx = Context(agent)

    # response = await agent.run("My name is Bob.", ctx=ctx)
    # print(f"\n{response}")
    # response = await agent.run("What was my name again?", ctx=ctx)
    # print(response)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    query_engine = index.as_query_engine(
        llm=llm, similarity_top_k=2
    )  # as shown in the Components in LlamaIndex section

    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="name",
        description="a specific description",
        return_direct=False,
    )
    query_engine_agent = AgentWorkflow.from_tools_or_functions(
        [query_engine_tool],
        llm=llm,
        system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. ",
    )

    response = await query_engine_agent.run("What is the weather like in New York?")
    print(response)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
