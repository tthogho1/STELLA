"""
Prompt assembly and the tool definitions built from an action map.

Memories accumulate across a task tree and the whole set is re-sent on every agent call,
so the caps here are what stops one oversized agent reply from overflowing the context
window. The ordering matters too: prompt caching only reuses a stable prefix.
"""
import re

from conftest import make_chat


def _agent(agent_id="a", name="A", description="d"):
    from stella_core.models.agent import Agent
    return Agent(agent_id=agent_id, name=name, short_description=description)


class TestPromptCaps:
    def test_only_the_newest_memories_are_kept(self):
        from stella_core.models.agent import Agent, MAX_MEMORY_ENTRIES

        built = Agent._construct_memory_string([f"entry{i}" for i in range(50)])

        assert built.count("entry") == MAX_MEMORY_ENTRIES
        assert "omitted" in built, "the reader should be told entries were dropped"
        assert "entry49" in built and "entry0\n" not in built

    def test_a_single_oversized_entry_is_truncated(self):
        from stella_core.models.agent import Agent, MAX_MEMORY_ENTRY_CHARS

        built = Agent._construct_memory_string(["x" * 50_000])

        assert len(built) < MAX_MEMORY_ENTRY_CHARS + 500
        assert "truncated" in built

    def test_chat_history_is_capped(self):
        from stella_core.models.agent import Agent, MAX_CHAT_HISTORY_MESSAGES

        built = Agent._construct_chat_string(make_chat([f"m{i}" for i in range(50)]))

        assert built.count("user: m") == MAX_CHAT_HISTORY_MESSAGES
        assert "m49" in built

    def test_short_input_is_left_alone(self):
        from stella_core.models.agent import Agent

        assert Agent._construct_memory_string([]) == ""
        built = Agent._construct_memory_string(["just one"])
        assert "just one" in built and "omitted" not in built


class TestPromptOrdering:
    def test_the_stable_chat_block_comes_before_the_volatile_memories(self):
        """The chat is fixed for a whole request while memories grow with every round,
        so the chat has to be the prefix for caching to reuse anything."""
        agent = _agent()
        built = (agent._construct_chat_string(make_chat(["hello"]))
                 + agent._construct_memory_string(["a memory"]))

        assert built.index("CHAT WITH THE USER") < built.index("AVAILABLE INFORMATION")


class TestToolDefinitions:
    def test_names_are_accepted_by_the_api(self):
        """OpenAI only accepts ^[a-zA-Z0-9_-]{1,64}$, and packaged agents use namespaced
        ids such as stella/stella_cactus_agent."""
        agent = _agent()
        action_map = {
            "1": _agent("demo_weather_agent", "WEATHER", "weather"),
            "2": _agent("stella/stella_cactus_agent", "Cactus", "a story"),
        }

        tools, _ = agent._build_tools(action_map)

        for tool in tools:
            assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tool["function"]["name"])

    def test_every_action_is_reachable_including_done(self):
        agent = _agent()
        action_map = {"1": _agent("one", "One", "d"), "2": _agent("two", "Two", "d")}

        tools, mapping = agent._build_tools(action_map)

        assert len(tools) == 3, "one tool per agent plus the done tool"
        assert sorted(mapping.values()) == ["0", "1", "2"]
        assert mapping[agent.done_tool_name] == agent.done_action

    def test_colliding_and_reserved_names_stay_distinct(self):
        """Two ids can sanitize to the same string, and an agent may be called
        respond_to_user, which is the name the done action uses."""
        agent = _agent()
        action_map = {
            "1": _agent("a/b", "A", "d"),
            "2": _agent("a-b", "B", "d"),
            "3": _agent("respond_to_user", "C", "d"),
        }

        tools, mapping = agent._build_tools(action_map)

        names = [t["function"]["name"] for t in tools]
        assert len(set(names)) == len(names), f"duplicate tool names: {names}"
        assert sorted(mapping.values()) == ["0", "1", "2", "3"]


class TestActionParsing:
    def test_an_index_is_pulled_out_of_a_wordy_reply(self):
        """The text protocol is the fallback when a model has no tool support, and models
        answer it with "1.", "Action 1" or a whole sentence."""
        from stella_core.models.agent import Agent

        action_map = {"1": _agent(), "2": _agent()}
        for reply in ("1", "1.", "Action 1", "I would pick action 1 here"):
            assert Agent._parse_action(reply, action_map) == "1"

    def test_an_unusable_reply_becomes_done(self):
        from stella_core.models.agent import Agent

        action_map = {"1": _agent()}
        for reply in (None, "", "no idea", "7"):
            assert Agent._parse_action(reply, action_map) == "0"
