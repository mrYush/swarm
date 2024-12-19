from pathlib import Path

import yaml

from swarm.repl import run_demo_loop
from agents import solver_agent, set_up_agents, decomposer_agent

if __name__ == "__main__":
    file_path = Path(__file__).resolve()
    config_path = file_path.parent / f"{file_path.stem}_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    client_config = {
        "base_url": config["base_url"]
    }
    set_up_agents(model_name=config["model_name"],
                  tool_choice=config["tool_choice"])
    run_demo_loop(
        starting_agent=decomposer_agent,
        stream=False,
        debug=False,
        client_config=client_config,
        possible_msg_keys=["role", "content"]
    )
