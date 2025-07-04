# https://huggingface.co/learn/agents-course/unit2/llama-index/workflows
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, Event, step, Context
from llama_index.utils.workflow import draw_all_possible_flows
import random


class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
    # @step
    # async def step_one(self, ev: StartEvent) -> ProcessingEvent:
    #     # Process initial data
    #     return ProcessingEvent(intermediate_result="Step 1 complete")

    # @step
    # async def step_two(self, ev: ProcessingEvent) -> StopEvent:
    #     # Use the intermediate result
    #     final_result = f"Finished processing: {ev.intermediate_result}"
    #     return StopEvent(result=final_result)

    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
      # store query in the context
      await ctx.set("query", "What is the capital of France?")

      # do something with context and event
      # val = ...

      # retrieve query from the context
      query = await ctx.get("query")

      return StopEvent(result=val)


async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    mw = MultiStepWorkflow(timeout=10, verbose=True)

    # result = await mw.run()
    # print(result)

    draw_all_possible_flows(mw)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
