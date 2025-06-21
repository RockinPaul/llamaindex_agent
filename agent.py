from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.evaluation import FaithfulnessEvaluator
from dotenv import load_dotenv
import chromadb
import os

load_dotenv()

openai_api_key = os.getenv("OPEN_AI_API_KEY")

llm = OpenAI(
    model_name="o4-mini-2025-04-16",
    temperature=0.7,
    max_tokens=100,
    api_key=openai_api_key,
)

# response = llm.complete("Hello, how are you?")
# print(response)


async def main():
    reader = SimpleDirectoryReader(input_dir="./data")
    documents = reader.load_data()

    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    db.delete_collection("alfred")
    chroma_collection = db.get_or_create_collection("alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=512, chunk_overlap=0),
            HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
        ],
        vector_store=vector_store,
    )

    nodes = await pipeline.arun(documents=documents)
    # print(nodes)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # refine: create and refine an answer by sequentially going through each retrieved text chunk. This makes a separate LLM call per Node/retrieved chunk.
    # compact (default): similar to refining but concatenating the chunks beforehand, resulting in fewer LLM calls.
    # tree_summarize: create a detailed answer by going through each retrieved text chunk and creating a tree structure of the answer.
    query_engine = index.as_query_engine(llm=llm, response_mode="compact")
    response = query_engine.query("Who is Paul?")

    print(response)

    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_result = evaluator.evaluate_response(response=response)

    print("---Evaluation---")
    print(f"Is Faithful: {eval_result.passing}")
    print(f"Reasoning: {eval_result.feedback}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
