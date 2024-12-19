import inspect
import json
import re
from datetime import datetime
from typing import Callable

from .types import Function, ChatCompletionMessageToolCall


def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }


def parse_tool_calls_from_content(
    content: str,
    debug: bool = False
) -> list[ChatCompletionMessageToolCall]:
    """
    Extracts tool calls from a string containing <tool_call> tags.
    Args:
        content: The content string to be parsed text between <tool_call> tags.
        debug: Whether to print debug information.
    Returns:
        A list of ChatCompletionMessageToolCall objects.
    """
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    tool_calls = []

    if content:
        matches = tool_call_pattern.findall(content)
        for match in matches:
            try:
                tool_call_data = json.loads(match)
                arg_str = json.dumps(tool_call_data.get("arguments", {}))
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=tool_call_data.get("id", "unknown"),
                        function=Function(
                            name=tool_call_data["name"],
                            arguments=arg_str
                        ),
                        type="function"
                    )
                )
            except json.JSONDecodeError as e:
                if debug:
                    print(f"Failed to parse tool call JSON: {e}")

    return tool_calls


def filter_args(args: dict, func: Callable) -> dict:
    """
    Filters the arguments dictionary to only include keys that are
    present in the function's parameter list.

    Args:
        args: The dictionary of arguments to be filtered.
        func: The function whose parameter list will be used for filtering.

    Returns:
        A dictionary containing only the arguments that are present in the
        function's parameter list.
    """
    expected_args = func.__code__.co_varnames[:func.__code__.co_argcount]
    return {k: v for k, v in args.items() if k in func.arguments}


def collect_messages_tail(
    messages: list[dict[str, str]],
    available_len: int = 512
) -> list[dict[str, str]]:
    """
    Collects the tail of messages that fit into the available length.
    Args:
        messages: A list of messages to be processed.
        available_len: The maximum length of the messages to be collected.
    Returns:
        A list of message texts that fit into the available length.
    """
    joint_length = 0
    index = len(messages) - 1
    for index in range(len(messages) - 1, -1, -1):
        msg_len = len(json.dumps(messages[index], ensure_ascii=False))
        # FIXME better count tokens than symbols
        if joint_length + msg_len > available_len:
            break
        joint_length += msg_len + 1
    messages_slice = messages[index:]
    if len(messages_slice) == 0:
        new_content = messages[-1]["content"][-available_len:]
        messages_slice = [
            {"role": messages[-1]["role"], "content": new_content}
        ]
    return [message for message in messages_slice]
