from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import asyncio
import os


# Define some tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    await increment_counter(ctx)
    return a + b


async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    await increment_counter(ctx)
    return a * b


async def increment_counter(ctx: Context):
    cur_state = await ctx.get("state")
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)


async def main():
    load_dotenv()

    openai_api_key = os.getenv("OPEN_AI_API_KEY")

    llm = OpenAI(
        model_name="o4-mini-2025-04-16",
        temperature=0.7,
        max_tokens=100,
        api_key=openai_api_key,
    )

    # we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
    multiply_agent = ReActAgent(
        name="multiply_agent",
        description="Is able to multiply two integers",
        system_prompt="You are a helpful assistant that can use a tool to multiply numbers. Tools usage is mandatory! If you don't have tools, delegate it to another agent.",
        tools=[multiply],
        tool_required=True,
        llm=llm,
    )

    addition_agent = ReActAgent(
        name="add_agent",
        description="Is able to add two integers",
        system_prompt="A helpful assistant that can use a tool to add numbers.",
        tools=[add],
        tool_required=True,
        llm=llm,
    )

    manager_prompt = """
You are a coordinator. Other available agents:
- add_agent: can add two integers
You MUST delegate if your own tools cannot solve the user request.
Use the format: Action: delegate_to=add_agent
"""

    manager = ReActAgent(
        name="manager",
        description="Routes tasks to the right specialist agent.",
        system_prompt=manager_prompt,
        allow_direct_tool_use=False,
        tool_required=True,
        tools=[],  # manager has none
        llm=llm,
    )

    # Create the workflow
    workflow = AgentWorkflow(
        agents=[manager, multiply_agent, addition_agent],
        root_agent="manager",
        initial_state={"num_fn_calls": 0},
        state_prompt="Current state: {state}. User message: {msg}",
    )

    # run the workflow with context
    ctx = Context(workflow)
    response = await workflow.run(
        user_msg="Use tools to add 5 and 3. Tools usage is mandatory!", ctx=ctx
    )
    print(response)

    # pull out and inspect the state
    state = await ctx.get("state")
    print(state["num_fn_calls"])


if __name__ == "__main__":
    asyncio.run(main())
