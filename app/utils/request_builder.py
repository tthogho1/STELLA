from app.models.chat import Chat
from app.openai_client import OpenAIClient, DEFAULT_RESPONSE_MODEL

import json
import os

# Ask the API to constrain the reply to JSON instead of relying on the prompt alone.
# Set USE_STRUCTURED_OUTPUTS="false" for a model that does not support response_format;
# build_request also falls back on its own if the API rejects the parameter.
USE_STRUCTURED_OUTPUTS = os.getenv('USE_STRUCTURED_OUTPUTS', 'true').strip().lower() not in ('false', '0', 'no')

# Envelope the model has to fill in. The payload itself is caller-defined, so the default
# only pins down the envelope; pass payload_schema to constrain the payload too.
_JSON_OBJECT_FORMAT = {"type": "json_object"}


def _strict_format(payload_schema: dict):
    """
    Builds a strict json_schema response format around a caller-supplied payload schema.
    Strict mode requires every property to be listed in "required" and
    "additionalProperties": false on every object, including inside payload_schema.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "api_request",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL including any query parameters"},
                    "json": payload_schema,
                },
                "required": ["url", "json"],
                "additionalProperties": False,
            },
        },
    }


class RequestBuilder:
    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client

    @staticmethod
    def _construct_chat_string(chat: Chat):
        chat_string = "=== CHAT WITH THE USER ===\n"
        for message in chat.chat_history:
            role = message['role']
            message = message['content']
            chat_string += f"{role}: {message}\n"
        chat_string += "=== (END OF CHAT) ===\n\n"
        return chat_string

    @staticmethod
    def _construct_memory_string(memories):
        if not memories:
            return ""
        memory_string = "=== AVAILABLE INFORMATION ===\n"
        for memory in memories:
            memory_string += f"{memory}"
        memory_string += "\n"
        return memory_string

    @staticmethod
    def _extract(result):
        """
        Pulls (url, json) out of a model reply. Structured outputs guarantee the reply is
        a JSON object, but not that the model filled in the envelope we asked for, so the
        keys are still read defensively.
        """
        json_data = json.loads(result)
        if not isinstance(json_data, dict):
            raise ValueError(f"Expected a JSON object, got {type(json_data).__name__}")
        return json_data.get("url", ""), json_data.get("json", {})

    def build_request(self, details=None, method=None, instructions=None, chat: Chat = None,
                      memories: list[str] = None, payload_schema: dict = None):
        """
        :param method: HTTP method to use (GET, POST, PUT, DELETE)
        :param details: Details about the API endpoint, including all RELEVANT information from the API Specification.
        :param instructions: List of instructions from the invoker that explains how to build the request
        :param chat: Chat object, used to see all messages in the chat
        :param memories: List of memories from the task
        :param payload_schema: Optional JSON Schema for the request body. When given, the
                               API guarantees the body matches it. Must satisfy strict-mode
                               rules (all properties required, additionalProperties false).
        :return: (url, json)
        """

        system_message = "You are a request builder.\n\nYou will be given:\n- a description of an API endpoint " \
                         "delimited by \"===\"- Unstructured information delimited by \"---\"\n- Additional instructions " \
                         "from the user\n\nCarefully examine the given information and create the filled in objects " \
                         "that should be sent to the api.\n\nUse the following structure for your answer:\n{\"url\": " \
                         "<full url including parameters>, \"json\": <filled json data>}\n\nYour answer should be " \
                         "loadable using Python's json.loads()."

        if method:
            system_message += f"\n\nThe request will be sent with the HTTP {method} method."

        user_message = f"API Endpoint:\n===\n{details}"
        user_message += "\n===\n\nUnstructured information:\n---\n"
        # Chat before memories: the chat is fixed for the whole request while memories
        # grow with every round, so this keeps the stable content at the front where
        # prompt caching can reuse it.
        if chat:
            user_message += self._construct_chat_string(chat)
        if memories:
            user_message += self._construct_memory_string(memories)

        if instructions:
            user_message += "---\n\nAdditional instructions:\n"
            user_message += instructions

        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        response_format = None
        if USE_STRUCTURED_OUTPUTS:
            response_format = _strict_format(payload_schema) if payload_schema else _JSON_OBJECT_FORMAT

        result = None
        if response_format is not None:
            try:
                result = self.openai_client.chat_completion(
                    messages=messages,
                    model=DEFAULT_RESPONSE_MODEL,
                    response_format=response_format,
                )
            except Exception as e:
                print(f"[REQUEST BUILDER] Structured output request failed "
                      f"({type(e).__name__}: {e}), retrying without response_format")
                result = None

        if result is None:
            result = self.openai_client.chat_completion(
                messages=messages,
                model=DEFAULT_RESPONSE_MODEL,
            )

        try:
            return self._extract(result)
        except Exception as e:
            raise Exception(f"Failed to parse response from OpenAI: {e}", result)
