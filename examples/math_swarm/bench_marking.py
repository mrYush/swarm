import hashlib
import logging
import shelve
from datetime import datetime
from pathlib import Path

import yaml
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from examples.math_swarm.agents import set_up_agents, solver_agent, \
    finalizing_agent
from examples.math_swarm.side_utils import IndentDumper
from swarm import Swarm


def get_cache(joint_hash):
    with shelve.open("cache.db") as db:
        return db.get(joint_hash)


def save_cache(joint_hash, data):
    with shelve.open("cache.db") as db:
        db[joint_hash] = data


if __name__ == "__main__":
    file_path = Path(__file__).resolve()
    config_path = file_path.parent / f"{file_path.stem}_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    dataset_name = config["dataset_name"]
    ds = load_dataset(dataset_name)
    test = ds["test"]
    random_samples = test.select(range(config["sample_cnt"]))
    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    set_up_agents(model_name=config["model_name"],
                  tool_choice=config["tool_choice"])
    agent = solver_agent
    agent.instructions += f"\nCreate solution for {config['max_turns']} turns."
    for_exam = []
    config_hash = hashlib.sha256(str(config).encode()).hexdigest()[:8]
    print(config_hash)
    for item in tqdm(random_samples):
        item_hash = hashlib.sha256(str(item).encode()).hexdigest()[:8]
        # print(item_hash)
        cache_data = get_cache(joint_hash=f"{item_hash}_{config_hash}")
        if cache_data:
            for_exam.append(cache_data)
            continue
        swarm = Swarm(
            client=client,
            possible_msg_keys=["role", "content"],
            closing_agent=finalizing_agent,
            max_symbol_cnt=config["max_symbol_cnt"],
        )

        history = [{"role": "user", "content": item["problem"]}]
        response = swarm.run(
            agent=agent,
            messages=history,
            context_variables={},
            stream=False,
            debug=True,
            max_turns=config["max_turns"],
        )
        data = {"response": response.model_dump(), "item": item}
        save_cache(joint_hash=f"{item_hash}_{config_hash}",
                   data=data)
        for_exam.append(data)
    output_folder = Path(config["output_folder"])

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_filename = f"example_responses_{config_hash}_{now}.yaml"
    with open(output_folder / dest_filename, "w") as iof:
        yaml.dump(data={"config": config, "results": for_exam}, stream=iof,
                  sort_keys=False, allow_unicode=True, Dumper=IndentDumper)
