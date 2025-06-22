from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
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

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")


def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)
# tool.call("New York")

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)
tool1 = QueryEngineTool.from_defaults(query_engine, name="some useful name", description="some useful description")
output = tool1.call("New York")
print(output)