"""
The LangChain tool wrapper.

The tool reports its own failures in an "error" field instead of raising, so the wrapper
has to turn those into something the coordinator can act on. A result that reads as empty
just makes the coordinator call the agent again.
"""
import json

import pytest

from conftest import make_chat

pytest.importorskip("langchain_core", reason="the wrapper needs langchain-core")


@pytest.fixture
def agent():
    from stella_agents.CustomAgents.CitySearchAgent import CitySearchAgent
    return CitySearchAgent()


class _Client:
    """Stands in for OpenAIClient, returning a fixed query extraction."""

    def __init__(self, reply="Osaka"):
        self.reply = reply
        self.calls = []

    def chat_completion(self, messages, model, response_format=None):
        self.calls.append(messages)
        return self.reply


def test_it_registers_as_a_leaf_agent(agent):
    """No connections to delegate to, so action selection is a wasted LLM call."""
    assert agent.agent_id == "city_search_agent"
    assert agent.skip_action_selection is True
    assert agent.forward_all_memory_entries_to_parent is True


def test_a_successful_search_is_summarised(agent, monkeypatch):
    payload = {
        "query": "Osaka", "count": 2,
        "results": [{"title": "Osaka", "url": "https://example.test/osaka", "content": "x" * 300},
                    {"title": "Kyoto", "url": "https://example.test/kyoto", "content": "y" * 300}],
        "summary": "Osaka is a lively port city known for its food.",
    }
    monkeypatch.setattr(
        "stella_agents.CustomAgents.CitySearchAgent.search_cities",
        type("T", (), {"invoke": staticmethod(lambda _: json.dumps(payload))})(),
    )

    out = agent.respond(_Client(), None, chat=make_chat(["where should I go in Japan?"]))

    assert "Osaka is a lively port city" in out
    assert "https://example.test/osaka" in out
    # The raw passages must not be carried up: memories are re-sent on every later call.
    assert "x" * 300 not in out


def test_a_tool_level_error_is_reported_not_swallowed(agent, monkeypatch):
    """Missing credentials come back as an "error" field, not an exception."""
    payload = {"query": "Osaka", "error": "missing env var(s): DATABRICKS_TOKEN", "results": []}
    monkeypatch.setattr(
        "stella_agents.CustomAgents.CitySearchAgent.search_cities",
        type("T", (), {"invoke": staticmethod(lambda _: json.dumps(payload))})(),
    )

    out = agent.respond(_Client(), None, chat=make_chat(["where should I go?"]))

    assert "DATABRICKS_TOKEN" in out
    assert "do not retry" in out, "otherwise the coordinator keeps calling the agent"


def test_no_results_tells_the_coordinator_to_stop(agent, monkeypatch):
    payload = {"query": "atlantis", "count": 0, "results": [], "summary": ""}
    monkeypatch.setattr(
        "stella_agents.CustomAgents.CitySearchAgent.search_cities",
        type("T", (), {"invoke": staticmethod(lambda _: json.dumps(payload))})(),
    )

    out = agent.respond(_Client(), None, chat=make_chat(["take me to atlantis"]))

    assert "found nothing" in out and "do not retry" in out


def test_an_exception_inside_langchain_does_not_escape(agent, monkeypatch):
    """An exception here would be caught by the worker, but it would fail the whole
    request instead of letting the coordinator answer without this agent."""
    def boom(_):
        raise RuntimeError("langchain blew up")

    monkeypatch.setattr(
        "stella_agents.CustomAgents.CitySearchAgent.search_cities",
        type("T", (), {"invoke": staticmethod(boom)})(),
    )

    out = agent.respond(_Client(), None, chat=make_chat(["where to?"]))

    assert "RuntimeError" in out and "do not retry" in out


def test_an_empty_query_never_reaches_the_tool(agent, monkeypatch):
    called = []
    monkeypatch.setattr(
        "stella_agents.CustomAgents.CitySearchAgent.search_cities",
        type("T", (), {"invoke": staticmethod(lambda q: called.append(q) or "{}")})(),
    )

    out = agent.respond(_Client(reply="  "), None, chat=make_chat(["hello"]))

    assert called == []
    assert "no query" in out


def test_malformed_tool_output_is_survived(agent, monkeypatch):
    monkeypatch.setattr(
        "stella_agents.CustomAgents.CitySearchAgent.search_cities",
        type("T", (), {"invoke": staticmethod(lambda _: "not json at all")})(),
    )

    out = agent.respond(_Client(), None, chat=make_chat(["where to?"]))

    assert "not json at all" in out


def test_the_tool_itself_reports_missing_credentials(monkeypatch):
    """Guards the contract the wrapper relies on: the tool returns, never raises."""
    from stella_agents.tools.city_search import search_cities

    monkeypatch.delenv("WORKSPACE_URL", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)

    data = json.loads(search_cities.invoke({"query": "Osaka"}))

    assert "missing env var" in data["error"]
    assert data["results"] == []


def test_a_successful_search_says_not_to_run_it_again(agent, monkeypatch):
    """
    The index is a similarity search with no notion of exclusion, so "coffee outside
    Africa" comes back with African cities and a summary saying they are not relevant.
    Left at that, the coordinator reads the request as unanswered and re-runs the same
    search until the depth limit ends the request with no answer.
    """
    payload = {
        "query": "best cities for coffee outside of Africa", "count": 3,
        "results": [{"title": "Coffee Bay", "url": "u1", "content": "c"},
                    {"title": "Addis Ababa", "url": "u2", "content": "c"},
                    {"title": "Harar", "url": "u3", "content": "c"}],
        "summary": "The provided information is not relevant to the query.",
    }
    monkeypatch.setattr(
        "stella_agents.CustomAgents.CitySearchAgent.search_cities",
        type("T", (), {"invoke": staticmethod(lambda _: json.dumps(payload))})(),
    )

    out = agent.respond(_Client(), None, chat=make_chat(["coffee, but not in Africa"]))

    assert "Addis Ababa" in out, "the results still have to be handed over"
    assert "do not call this agent again" in out
    assert "same rows" in out, "the coordinator needs to know a retry changes nothing"
