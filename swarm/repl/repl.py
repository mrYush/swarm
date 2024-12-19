import json

from openai import OpenAI

from swarm import Swarm, Agent


def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]


def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


def run_demo_loop(
    starting_agent, context_variables=None, stream=False, debug=False,
    client_config: dict | None = None,
    possible_msg_keys: list[str] | None = None,
    closing_agent: Agent | None = None
) -> None:
    """
    Run a simple REPL loop for interacting with a Swarm agent.
    Parameters
    ----------
    starting_agent
    context_variables
    stream
    debug
    client_config: dict, optional
        Configuration for the OpenAI client.
        Example:
        >>> client_config = {
        ...     "base_url": "https://api.openai.com/v1",
        ...     "api_key": "your-api-key",
        ...     "organization_id": "your-org-id",
        ...     "http_client": httpx.Client(proxies="list of proxies"),
        ... }
    possible_msg_keys: list[str], optional
        List of possible keys for messages. For making compatible with
        different API.
        Example:
        >>> possible_msg_keys = ["role", "content", "tool_calls"]
    closing_agent: Agent, optional
        Agent to close the conversation.
    Returns
    -------

    """
    if client_config is None:
        client_config = {}
    client = OpenAI(**client_config)
    swarm = Swarm(
        client=client,
        possible_msg_keys=possible_msg_keys,
        closing_agent=closing_agent
    )
    print("Starting Swarm CLI 🐝")

    messages = []
    agent = starting_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})

        response = swarm.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent = response.agent
