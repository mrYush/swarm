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
        "integrate them, and produce the final solution"
    ),
    functions=[],
)


solver_agent = Agent(
    name="Math Solver Agent",
    instructions="Generate solutions by following the provided plan step-by-step",
    functions=[],
)


def launch_solver_agent(solution_plan: str):
    """Launch a new agent to solve a math problem."""
    global solver_agent
    solver_agent = copy.deepcopy(solver_agent)
    solver_agent.instructions += f"\n\n{solution_plan}"
    return solver_agent


decomposer_agent = Agent(
    name="Math Decomposer Agent",
    instructions=(
        "Analyze the MATH-500 task, decompose it into 5 subtasks, "
        "and assign each subtask to a different agent. For each subtask "
        "launch a new agent through the 'launch_solver_agent' function."
    ),
    functions=[launch_solver_agent]
)

def transfer_to_decomposer():
    return decomposer_agent


def set_up_agents(model_name: str, tool_choice: str = None):
    solver_agent.model = model_name
    decomposer_agent.model = model_name
    if tool_choice:
        solver_agent.tool_choice = tool_choice
        decomposer_agent.tool_choice = tool_choice
    solver_agent.functions.append(transfer_to_decomposer)
