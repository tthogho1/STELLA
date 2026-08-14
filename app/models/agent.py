import os
import re

from dotenv import load_dotenv

from app.models.chat import Chat
from app.openai_client import OpenAIClient, DEFAULT_ACTION_SELECTION_MODEL, DEFAULT_RESPONSE_MODEL
from app.utils.request_builder import RequestBuilder
# Load the environment variables from the .env file
load_dotenv()

# Action selection asks the model to pick a tool instead of writing an index as free text.
# Set USE_TOOL_CALLING="false" to go back to the text protocol (e.g. for a model without
# tool support); select_action also falls back on its own if a tool call raises.
USE_TOOL_CALLING = os.getenv('USE_TOOL_CALLING', 'true').strip().lower() not in ('false', '0', 'no')

# When the model asks for several agents in one turn, run them as siblings instead of
# keeping only the first. Requires tool calling. Set PARALLEL_AGENT_CALLS="false" to keep
# the one-agent-at-a-time behaviour.
PARALLEL_AGENT_CALLS = os.getenv('PARALLEL_AGENT_CALLS', 'true').strip().lower() not in ('false', '0', 'no')


class Agent:
    """
    Loads and stores information about an agent
    """
    default_system_action_selection_instructions = "\n\n==== INSTRUCTIONS ====\n" \
                                                   "Carefully examine the chat history. " \
                                                   "Focus on the user's latest request and " \
                                                   "select the most relevant action to help solve the user's request." \
                                                   "Only output the index of the action you want to use. Do not " \
                                                   "output any other text except the number. " \
                                                   "\n\nCarefully analyze the provided details from your previous " \
                                                   "actions. If there is enough information to " \
                                                   "answer the user's request, or if no available actions are" \
                                                   " relevant to help solve the user's request, respond with '0'. "
    default_tool_action_selection_instructions = \
        "\n\n==== INSTRUCTIONS ====\n" \
        "Carefully examine the chat history and focus on the user's latest request.\n\n" \
        "First read AVAILABLE INFORMATION. It holds the results of the actions already " \
        "taken for this request. Anything answered there is done: never call a tool to " \
        "fetch information you have already been given.\n\n" \
        "If AVAILABLE INFORMATION already covers the whole request, or none of the tools " \
        "are relevant to it, respond to the user instead of calling anything else.\n\n" \
        "Otherwise call one tool for each part of the request that is still unanswered. " \
        "Independent parts are handled at the same time, so call them together in this " \
        "one turn rather than one at a time. Call each tool at most once, and do not " \
        "combine responding to the user with any other tool."
    default_system_response_instructions = "\n\n==== INSTRUCTIONS ====\n" \
                                           "Generate an appropriate response to the user's request using the " \
                                           "provided details. Focus on the user's latest message, and use the " \
                                           "provided details to respond in a proper manner. Don't lie or make " \
                                           "things up. Be as helpful, nice and concise as possible."
    default_done_condition = "If the user's task has been completed and you want to generate a response"

    # The action key that means "stop delegating and generate a response".
    done_action = "0"
    # Tool the model calls for the "done" action. Agent ids that sanitize to this are suffixed.
    done_tool_name = "respond_to_user"
    # OpenAI only accepts tool names matching ^[a-zA-Z0-9_-]{1,64}$
    _invalid_tool_name_chars = re.compile(r'[^a-zA-Z0-9_-]')

    def __init__(
            self,
            agent_id: str,
            name: str,
            short_description: str,
            model_for_action_selection: str = None,  # Defaults to OPENAI_MODEL_ACTION_SELECTION
            model_for_response: str = None,  # Defaults to OPENAI_MODEL_RESPONSE
            wants_chat: bool = True,
            wants_memories: bool = True,
            forward_all_memory_entries_to_parent: bool = False,
            forward_last_memory_to_parent: bool = False,
            system_action_selection_instructions: str = default_system_action_selection_instructions,
            system_response_instructions: str = default_system_response_instructions,
            done_condition: str = default_done_condition,
            skip_response: bool = False,
            skip_action_selection: bool = False,
            connections_forced: dict = None,
            connections_available: dict = None,
            max_depth: int = None,
            on_completion: callable = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.short_description = short_description
        self.model_for_action_selection = model_for_action_selection or DEFAULT_ACTION_SELECTION_MODEL
        self.model_for_response = model_for_response or DEFAULT_RESPONSE_MODEL
        self.wants_chat = wants_chat
        self.wants_memories = wants_memories
        self.forward_all_memory_entries_to_parent = forward_all_memory_entries_to_parent
        self.forward_last_memory_to_parent = forward_last_memory_to_parent
        self.system_action_selection_instructions = system_action_selection_instructions
        self.system_response_instructions = system_response_instructions
        self.done_condition = done_condition
        self.skip_response = skip_response
        self.skip_action_selection = skip_action_selection
        self.connections_forced = connections_forced
        self.connections_available = connections_available
        self.max_depth = max_depth if max_depth is not None else int(os.getenv('AGENT_MAX_DEPTH', 99999999))
        self.on_completion = on_completion

        if self.connections_forced is None:
            self.connections_forced = {}
        if self.connections_available is None:
            self.connections_available = {}

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
    def _construct_chat_string(chat: Chat):
        chat_string = "=== CHAT WITH THE USER ===\n"
        for message in chat.chat_history:
            role = message['role']
            message = message['content']
            chat_string += f"{role}: {message}\n"
        chat_string += "=== (END OF CHAT) ===\n\n"
        return chat_string

    @staticmethod
    def _parse_action(raw_action, action_map: dict):
        """
        The model is told to reply with nothing but an index, but it regularly answers with
        "1.", "Action 1" or a whole sentence. Pull the first integer out of the reply and
        fall back to "0" (done) when there is nothing usable, so that a malformed answer
        ends the task cleanly instead of raising a KeyError on the action map.
        :param raw_action: The raw reply from the model
        :param action_map: The action map the reply has to be valid for
        :return: A key that is guaranteed to be "0" or present in the action map
        """
        if raw_action is None:
            print("[AGENT] Action selection returned nothing, defaulting to 0 (done)")
            return "0"

        match = re.search(r'\d+', str(raw_action))
        if not match:
            print(f"[AGENT] Could not parse an action from '{raw_action}', defaulting to 0 (done)")
            return "0"

        action = match.group()
        if action != "0" and action not in action_map:
            print(f"[AGENT] Action '{action}' is not in the action map, defaulting to 0 (done)")
            return "0"

        return action

    def _construct_available_actions_string(self, action_map: dict):
        available_actions_string = "=== AVAILABLE ACTIONS ===\n"
        available_actions_string += f"0: Done - Respond to the user ({self.done_condition})\n"
        for action_id in action_map.keys():
            available_actions_string += f"{action_id}: {action_map[action_id].short_description}\n"
        available_actions_string += "=== (END OF AVAILABLE ACTIONS) ===\n"
        return available_actions_string

    @classmethod
    def _tool_name_for(cls, agent, action_id: str, taken: set):
        """
        Derives a tool name the API will accept from an agent id. Agent ids may contain
        characters OpenAI rejects (namespaced ids such as "stella/stella_cactus_agent"),
        and two agents could sanitize down to the same name, so collisions and the
        reserved "done" name fall back to a suffix that is unique by construction.
        """
        base = cls._invalid_tool_name_chars.sub('_', agent.agent_id or '').strip('_')[:56]
        if not base:
            base = f"agent_{action_id}"
        name = base
        if name == cls.done_tool_name or name in taken:
            name = f"{base}_{action_id}"
        return name

    def _build_tools(self, action_map: dict):
        """
        Turns the action map into tool definitions. Returns the tools plus a mapping from
        tool name back to the action key, because the rest of the task loop still speaks
        in action keys ("0" or a key of action_map).
        :param action_map: The action map to expose
        :return: (tools, tool_name_to_action)
        """
        tools = [{
            "type": "function",
            "function": {
                "name": self.done_tool_name,
                "description": f"Done - respond to the user. {self.done_condition}.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }]
        tool_name_to_action = {self.done_tool_name: self.done_action}

        for action_id, agent in action_map.items():
            name = self._tool_name_for(agent, action_id, set(tool_name_to_action))
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{agent.name}: {agent.short_description}",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            })
            tool_name_to_action[name] = action_id

        return tools, tool_name_to_action

    def _action_selection_system_message(self, use_tools: bool):
        """
        The default instructions tell the model to answer with a bare index, which is wrong
        for the tool path. Use the tool-specific defaults there, but keep any instructions
        an agent supplied itself so custom guidance is not silently dropped.
        """
        if not use_tools:
            return self.system_action_selection_instructions

        instructions = self.default_tool_action_selection_instructions
        if self.system_action_selection_instructions != self.default_system_action_selection_instructions:
            instructions += "\n\n" + self.system_action_selection_instructions
        return instructions

    def select_action(self, openai_client: OpenAIClient, action_map: dict, chat: Chat = None, memories=None):
        """
        Uses the Action Map to select an action.

        Each available agent is exposed as a tool, so the model returns a structured tool
        call rather than free text that has to be parsed for an index. Falls back to the
        legacy text protocol if tool calling is disabled or the model does not support it.
        :param openai_client:
        :param action_map:
        :param chat:
        :param memories:
        :return: "0" (done) or a key of action_map
        """
        if USE_TOOL_CALLING:
            try:
                return self._select_actions_with_tools(openai_client, action_map, chat, memories)[0]
            except Exception as e:
                print(f"[AGENT] Tool-based action selection failed ({type(e).__name__}: {e}), "
                      f"falling back to text-based selection")

        return self._select_action_with_text(openai_client, action_map, chat, memories)

    def select_actions(self, openai_client: OpenAIClient, action_map: dict, chat: Chat = None, memories=None):
        """
        Like select_action, but may return several actions to run as siblings.

        The model can ask for more than one agent in a single turn, and the task tree can
        carry them as parallel children. Subclasses that implement their own select_action
        keep full control: their single choice is honoured as-is.
        :return: A list of action keys. ["0"] means done. Never empty.
        """
        # An agent with hand-written selection logic decides on its own.
        if type(self).select_action is not Agent.select_action:
            return [str(self.select_action(openai_client, action_map, chat, memories))]

        if USE_TOOL_CALLING:
            try:
                actions = self._select_actions_with_tools(openai_client, action_map, chat, memories)
                if actions:
                    return actions
            except Exception as e:
                print(f"[AGENT] Tool-based action selection failed ({type(e).__name__}: {e}), "
                      f"falling back to text-based selection")

        return [self._select_action_with_text(openai_client, action_map, chat, memories)]

    def _select_actions_with_tools(self, openai_client: OpenAIClient, action_map: dict, chat: Chat = None,
                                   memories=None):
        tool_calls, tool_name_to_action = self._request_tool_selection(
            openai_client, action_map, chat, memories)

        if not tool_calls:
            print(f"[AGENT] {self.name} returned no tool call, defaulting to 0 (done)")
            return [self.done_action]

        actions = []
        for call in tool_calls:
            action = tool_name_to_action.get(call.function.name)
            if action is None:
                print(f"[AGENT] {self.name} called unknown tool '{call.function.name}', ignoring it")
                continue
            # The model sometimes asks for the same agent twice; once is enough.
            if action not in actions:
                actions.append(action)

        if not actions:
            return [self.done_action]

        # "Done" alongside real work means the model both delegated and tried to finish.
        # The delegation wins; the parent gets another turn once the children report back.
        if len(actions) > 1 and self.done_action in actions:
            actions.remove(self.done_action)

        if not PARALLEL_AGENT_CALLS:
            return actions[:1]

        return actions

    def _request_tool_selection(self, openai_client: OpenAIClient, action_map: dict, chat: Chat = None,
                                memories=None):
        """
        Runs the tool-calling request and returns (tool_calls, tool_name_to_action).
        """
        tools, tool_name_to_action = self._build_tools(action_map)

        # The actions no longer have to be serialized into the prompt: they are the tools.
        user_message = f"{self._construct_memory_string(memories) if memories else ''}" \
                       f"{self._construct_chat_string(chat) if chat else ''}"

        messages = [
            {
                "role": "system",
                "content": self._action_selection_system_message(use_tools=True)
            },
            {
                "role": "user",
                "content": user_message or "Select the most relevant tool."
            }
        ]

        message = openai_client.tool_call(
            messages=messages,
            model=self.model_for_action_selection,
            tools=tools,
            tool_choice="required",
        )

        # tool_choice="required" should always produce a call, but an empty list is
        # handled by the callers rather than trusted.
        return getattr(message, "tool_calls", None) or [], tool_name_to_action

    def _select_action_with_text(self, openai_client: OpenAIClient, action_map: dict, chat: Chat = None,
                                 memories=None):
        system_message = self.system_action_selection_instructions

        user_message = f"{self._construct_memory_string(memories) if memories else ''}" \
                       f"{self._construct_chat_string(chat) if chat else ''}" \
                       f"{self._construct_available_actions_string(action_map)}" \
                       f"{self.system_action_selection_instructions}"

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

        action_selected = openai_client.chat_completion(
            messages=messages,
            model=self.model_for_action_selection,
        )

        return self._parse_action(action_selected, action_map)


    def respond(self, openai_client: OpenAIClient, request_builder: RequestBuilder, chat: Chat = None, memories=None):
        system_message = self.system_response_instructions

        user_message = f"{self._construct_memory_string(memories) if memories else ''}" \
                       f"{self._construct_chat_string(chat) if chat else ''}" \
                       f"{self.system_response_instructions}"

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

        response_generated = openai_client.chat_completion(
            messages=messages,
            model=self.model_for_response,
        )

        return response_generated
