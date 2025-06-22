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
tool1 = QueryEngineTool.from_defaults(
    query_engine, name="some useful name", description="some useful description"
)
output = tool1.call("New York")
print(output)


# Load the toolspec and convert it to a list of tools.
# from llama_index.tools.google import GmailToolSpec

# tool_spec = GmailToolSpec()
# tool_spec_list = tool_spec.to_tool_list()
# print("--- Gmail Tools ---")
# for tool in tool_spec_list:
#     print(f"Name: {tool.metadata.name}")
#     print(f"Description: {tool.metadata.description}")
#     print("--------------------")

# MCP tools
from llama_index.tools.mcp import MCPToolSpec, BasicMCPClient


async def get_agent(tool_spec: MCPToolSpec) -> Agent:
    # We consider there is a mcp server running on 127.0.0.1:8000, or you can use the mcp client to connect to your own mcp server.
    mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")
    mcp_tool = McpToolSpec(client=mcp_client)

    # get the agent
    agent = await get_agent(mcp_tool)

    # create the agent context
    agent_context = Context(agent)
    return agent_context


# OnDemandToolLoader: This tool turns any existing LlamaIndex data loader (BaseReader class) into 
# a tool that an agent can use. The tool can be called with all the parameters needed to trigger 
# load_data from the data loader, along with a natural language query string. 
# During execution, we first load data from the data loader, 
# index it (for instance with a vector store), and then query it ‘on-demand’. 
# All three of these steps happen in a single tool call.

# LoadAndSearchToolSpec: The LoadAndSearchToolSpec takes in any existing Tool as input. 
# As a tool spec, it implements to_tool_list, and when that function is called, 
# two tools are returned: a loading tool and then a search tool. 
# The load Tool execution would call the underlying Tool, and then index the output 
# (by default with a vector index). The search Tool execution would take in a query 
# string as input and call the underlying index.
