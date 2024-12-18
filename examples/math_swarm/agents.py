"""
Each agent will get a task to solve.
Agent can split the task into sub-tasks and delegate them to other agents.
Each subtask can be solved by a different agent.
All solutions will be collected and the best solution will be chosen.
"""
from swarm import Agent


def solve_math_problem() -> str:
    """Solve a math problem."""
    print(f"[mock] Solving math problem: ...")
    return "The answer is 42"


solver_agent = Agent(
    name="Math Solver Agent",
    instructions="Solve the math problem.",
    functions=[solve_math_problem],
)
