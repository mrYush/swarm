"""
Each agent will get a task to solve.
Agent can split the task into sub-tasks and delegate them to other agents.
Each subtask can be solved by a different agent.
All solutions will be collected and the best solution will be chosen.
"""
import copy

from swarm import Agent

FINAL_PROMPT = """
Collect all the results from the agents, combine them, and generate the final solution. 
Return the result as JSON with the key 'answer'. 
For example, for the task "2 + 4 + 19", the expected output is: {"answer": 25}.
""".strip()
finalizing_agent = Agent(
    name="Finalising Agent",
    instructions=FINAL_PROMPT,
    functions=[],
    temperature=0.2,
)


def finalize():
    return finalizing_agent


solver_agent = Agent(
    name="Math Solver Agent",
    instructions=(
        "You have to prepare new step-by-step solutions for the math problem and solve it. "
        "After that, you should use  'launch_solver_agent' tool five times to generate all possible solutions. "
        "If all possible solutions are generated, use 'finalize' tool to complete the process."
    ),
    functions=[finalize],
    temperature=0.2
)


check_agent = Agent(
    name="Checker",
    instructions="User sends a json with keys: 'problem', 'answer', 'true_answer'. You shoul return json with key 'correct' and value 'true' or 'false'.",
    functions=[],
    temperature=0
)


def launch_solver_agent():
    """
    If any args or kwargs are passed, they will be propagated to the next agent.
    """
    return solver_agent


def set_up_agents(model_name: str, tool_choice: str = None):
    solver_agent.model = model_name
    finalizing_agent.model = model_name
    check_agent.model = model_name
    if tool_choice:
        solver_agent.tool_choice = tool_choice
        finalizing_agent.tool_choice = tool_choice
    solver_agent.functions.append(launch_solver_agent)
