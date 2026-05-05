import json

import litellm

from lab.settings import Settings

SYSTEM_PROMPT = """You are a helpful assistant. Please use the provided tools to answer the user's question."""
USER_PROMPT = """How many championships has Lewis Hamilton won?"""


settings = Settings()
if not settings.litellm_api_key:
    raise SystemExit("Please set LITELLM_API_KEY to your LiteLLM API key.")

litellm.api_base = settings.litellm_base_url
litellm.api_key = settings.litellm_api_key

def _completion_kwargs():
    return {
        "model": settings.litellm_model,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "presence_penalty": settings.presence_penalty,
        "repetition_penalty": settings.repetition_penalty,
        "extra_body": {  # Please disable thinking to get responses in a reasonable time frame
            "chat_template_kwargs": {"enable_thinking": False}
        },
    }

def get_championships(driver_name) -> int:
    """Get the number of championships won by a Formula 1 driver.

    Arguments:
        driver_name: The name of the Formula 1 driver.

    Returns:
        The number of championships won by the driver, or 0 if the driver is not in the list.
    """
    championships = {
        "Michael Schumacher": 7,
        "Lewis Hamilton": 7,
        "Juan Manuel Fangio": 5,
        "Alain Prost": 4,
        "Sebastian Vettel": 4,
        "Ayrton Senna": 3,
        "Nelson Piquet": 3,
        "Niki Lauda": 3,
        "Jackie Stewart": 3,
        "Jim Clark": 2,
    }

    return championships.get(driver_name, 0)


functions = {
    "get_championships": get_championships,
}

def main():
    messages: list[litellm.Message | dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    # Populate the TOOLS variable with the available functions using explicit JSON schemas.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_championships",
                "description": get_championships.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "driver_name": {
                            "type": "string",
                            "description": "The Formula 1 driver name.",
                        }
                    },
                    "required": ["driver_name"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    response = litellm.completion(
        **_completion_kwargs(),
        messages=messages,
        tools=tools,
        # tool_choice="required" # Uncomment this line to require the model to use at least one tool
        stream=False,  # Streaming is not supported with tool calls
    )

    # Check the response type to ensure it's not a stream
    if not isinstance(response, litellm.ModelResponse):
        raise ValueError(
            "Tool calls are not supported in streaming mode. Please disable streaming."
        )

    response_message = response.choices[0].message

    # Process tool calls if there are any.
    # The model may choose to call multiple tools, so we loop until there are no more tool calls in the response.
    tool_calls = response_message.tool_calls or []
    for tool_call in tool_calls:
        # Add the LLM response to the messages list before processing the tool call
        messages.append(response_message)

        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        if not function_name or function_name not in functions:
            continue

        # We want to pass exceptions to the model
        function_response = ""
        try:
            function_response = str(functions[function_name](**function_args))
        except Exception as e:
            function_response = str(e)
        
        print(f"{function_name}({function_args}) -> {function_response}")

        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )

    # After processing all tool calls, we send a final completion to get the model's response based on the tool outputs.
    response = litellm.completion(
        **_completion_kwargs(),
        messages=messages,
        stream=True,  # We can enable streaming for the final response
    )

     # If streaming is disabled, the response will be a ModelResponse object. 
    if isinstance(response, litellm.ModelResponse):
        print(response.choices[0].message.content)
        return

    # If streaming is enabled, it will be an iterator of ModelResponse objects.
    for chunk in response:
        print(chunk.choices[0].delta.content or "", end="", flush=True)


if __name__ == "__main__":
    main()
