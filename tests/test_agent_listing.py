"""
Listing the agents a server can add.

/add takes an agent_id and nothing told you which ones exist -- /status shows only what a
workspace already has. Without a listing the ids had to come from the README or from
reading stella_agents/.
"""
import types

import pytest


@pytest.fixture
def agents():
    from stella_core.models.agent import Agent
    return [
        Agent(agent_id="b_agent", name="B", short_description="second alphabetically"),
        Agent(agent_id="a_agent", name="A", short_description="first alphabetically",
              connections_available={"b_agent": {}}),
    ]


@pytest.fixture
def listing(agents):
    """Calls the view with a stubbed agent storage and returns the parsed body."""
    from flask import Flask
    from app.views.agent import list_agents

    app = Flask(__name__)
    app.extensions['agent_storage'] = agents          # the view only iterates it
    with app.app_context():
        body, status = list_agents.__wrapped__() if hasattr(list_agents, "__wrapped__") \
            else list_agents()
        return body.get_json(), status


def test_every_loaded_agent_is_listed(listing):
    body, status = listing

    assert status == 200
    assert body["count"] == 2
    assert {a["agent_id"] for a in body["agents"]} == {"a_agent", "b_agent"}


def test_the_order_is_stable(listing):
    """Sorted by id, so the list does not reshuffle between calls."""
    body, _ = listing

    assert [a["agent_id"] for a in body["agents"]] == ["a_agent", "b_agent"]


def test_each_entry_carries_what_you_need_to_choose(listing):
    body, _ = listing
    entry = next(a for a in body["agents"] if a["agent_id"] == "a_agent")

    assert entry["name"] == "A"
    assert entry["short_description"] == "first alphabetically"


def test_a_middle_tier_agent_shows_what_it_delegates_to(listing):
    """connections_available is the only place the tree below an agent is visible."""
    body, _ = listing
    entry = next(a for a in body["agents"] if a["agent_id"] == "a_agent")

    assert entry["delegates_to"] == ["b_agent"]


def test_a_leaf_delegates_to_nothing(listing):
    body, _ = listing
    entry = next(a for a in body["agents"] if a["agent_id"] == "b_agent")

    assert entry["delegates_to"] == []
