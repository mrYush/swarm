import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI
from tqdm import tqdm

from examples.math_swarm.agents import set_up_agents, check_agent
from examples.math_swarm.side_utils import IndentDumper
from swarm import Swarm

LOGGER = logging.getLogger()


def proceed(results: list[dict], agent) -> dict:
    scores = {}
    verdicts = []
    for result in tqdm(results):
        raw_response = result["response"]
        item = result["item"]
        item_hash = hashlib.sha256(str(result).encode()).hexdigest()[:8]
        if item_hash in scores:
            LOGGER.warning(f"Item {item_hash} already in scores")
            continue
        problem = item["problem"]
        messages = raw_response["messages"]
        answer = messages[-1]["content"]
        true_answer = item["answer"]
        client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
        swarm = Swarm(
            client=client,
            possible_msg_keys=["role", "content"],
            closing_agent=None,
            max_symbol_cnt=config["max_symbol_cnt"],
        )
        content = json.dumps({
            "problem": problem,
            "answer": answer,
            "true_answer": true_answer
        })
        history = [{"role": "user", "content": content}]
        response = swarm.run(
            agent=agent,
            messages=history,
            context_variables={},
            stream=False,
            debug=False,
            max_turns=1,
        )
        new_row = result.copy()
        ans_content = response.messages[0]["content"]
        new_row["verdict"] = json.loads(ans_content)["correct"]
        new_row["item_hash"] = item_hash
        scores[item_hash] = new_row
    return scores


def process_file(file_path: Path, agent) -> dict:
    with open(file_path) as iof:
        data = yaml.safe_load(iof)
    launch_config = data["config"]
    print(yaml.dump(launch_config, allow_unicode=True, sort_keys=False))
    file_scores = proceed(results=data["results"], agent=agent)
    dest_file = file_path.with_name(file_path.stem + "_scores.yaml")
    with open(dest_file, "w") as iof:
        yaml.dump(file_scores, iof, allow_unicode=True, sort_keys=False,
                  Dumper=IndentDumper)
    score_df = pd.DataFrame(file_scores).T
    # score_df.to_csv(dest_file.with_suffix(".csv"))
    launch_config["accuracy"] = float((score_df["verdict"] == "true").mean())
    return launch_config


if __name__ == "__main__":
    LOGGER.setLevel(logging.INFO)
    file_path = Path(__file__).resolve()
    config_path = file_path.parent / f"{file_path.stem}_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    client_config = {"base_url": config["base_url"]}
    set_up_agents(model_name=config["model_name"],
                  tool_choice=config["tool_choice"])
    agent = check_agent
    folder = Path(config["data_folder"])
    final_results = {}
    for filename in config["file_names"]:
        file_path = folder / filename
        next_row = process_file(file_path=file_path, agent=agent)
        final_results[filename] = next_row
    fdf = pd.DataFrame(final_results).T
    fdf.to_csv(folder / "final_results.csv")
    LOGGER.info("Done")
