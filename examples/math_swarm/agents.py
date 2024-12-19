"""
Each agent will get a task to solve.
Agent can split the task into sub-tasks and delegate them to other agents.
Each subtask can be solved by a different agent.
All solutions will be collected and the best solution will be chosen.
"""
import copy

from swarm import Agent


finalizer_agent = Agent(
    name="Finalizer Agent",
    instructions=(
        "Collect all the results from the agents, "
        "integrate them, and produce the final solution."
        "It should be only short answer."
    ),
    functions=[],
    temperature=0.0,
)


def finalize():
    return finalizer_agent


solver_agent = Agent(
    name="Math Solver Agent",
    instructions=(
        "Prepare new step-by-step solutions for the math problem and solve it. "
        "If fewer than 5 solutions are proposed, you must call 'launch_solver_agent'. "
        "In all cases, after generating solutions, ensure to call 'finalize' to complete the process."
    ),
    functions=[finalize],
    temperature=0.0
)


def launch_solver_agent():
    return solver_agent


def set_up_agents(model_name: str, tool_choice: str = None):
    solver_agent.model = model_name
    finalizer_agent.model = model_name
    if tool_choice:
        solver_agent.tool_choice = tool_choice
        finalizer_agent.tool_choice = tool_choice
    solver_agent.functions.append(launch_solver_agent)
