from llama_index.core.workflow import StartEvent, StopEvent, Workflow, Event, step
import random

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")

class ProcessingEvent(Event):
    intermediate_result: str
  
class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")
    
    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


async def main():
    w = MyWorkflow(timeout=10, verbose=False)
    mw = MultiStepWorkflow(timeout=10, verbose=True)

    result = await mw.run()
    print(result)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())