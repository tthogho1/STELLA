"""
Driving the agents directly, with no server running.

stella_core is the agent runtime and imports no web framework, so a request can be run
from an ordinary Python process: build the pieces the server would normally build, create
a task, then keep calling Task.execute() with whatever it hands back. The queue in
TaskManager is a thread pool around exactly this loop.

    cd app && python ../examples/run_without_a_server.py "What is the weather in Kyoto?"

Run it from app/ so that load_dotenv() finds app/.env, the same reason `stella serve`
does. Everything the server does that is *not* needed here -- Flask, SocketIO, JWT, the
CLI -- is simply absent.
"""
import os
import sys

from dotenv import find_dotenv, load_dotenv

# usecwd=True because load_dotenv() otherwise searches upward from *this file*, and this
# file is in examples/ -- it would never find app/.env. Everything else in the project
# resolves .env against the working directory too, which is why the server chdirs first.
load_dotenv(dotenv_path=find_dotenv(usecwd=True))

import stella_agents
from stella_core.agent_storage import AgentStorage
from stella_core.db import db, init_database
from stella_core.events import CollectingSink
from stella_core.models.task import Task, configure_default_agents
from stella_core.openai_client import OpenAIClient
from stella_core.trace import build_trace, render_trace
from stella_core.utils.request_builder import RequestBuilder

# Which agents this request may use. Any agent_id under stella_agents/ works.
AGENTS = ["demo_weather_agent", "brewery_agent"]

# A tree could in principle keep delegating; stop rather than loop forever.
MAX_STEPS = 40


def build_runtime():
    """The four things Task.execute() needs. app/server.py builds these too."""
    # The database is initialised explicitly -- importing stella_core.db no longer picks
    # a backend as a side effect, so a library user decides when (and which).
    init_database()

    agents_dir = os.path.dirname(os.path.abspath(stella_agents.__file__))
    storage = AgentStorage(agent_dirs=[agents_dir])

    # Which agent leads when a workspace has agents, and which greets an empty one.
    # The runtime has no built-in ids; the host application supplies them.
    configure_default_agents(
        general_agent_id='stella_coordinator_agent',
        empty_workspace_agent_id='stella_welcome_agent',
    )

    client = OpenAIClient()
    return storage, client, RequestBuilder(openai_client=client)


def run(message, agents=AGENTS):
    """Runs one request to completion and returns (answer, sink, top_level_task_id)."""
    storage, client, request_builder = build_runtime()

    # A chat is the unit a request belongs to, so one is still needed -- it is where the
    # history lives and where the answer is recorded.
    user = db.create_user(f"local-{os.urandom(4).hex()}", b"not-a-real-password")
    workspace = db.create_workspace(user.id, "local", {a: {} for a in agents})
    chat = db.create_chat(workspace.id, user.id)
    chat.add_message(role="user", content=message)
    db.update_chat(chat)

    task = Task.create_top_level_task(db.get_chat_by_id(chat.chat_id))

    # Instead of SocketIO, collect what the runtime emits. Any object with an emit()
    # would do; see stella_core/events.py.
    sink = CollectingSink()

    pending = [task.task_id]
    for _ in range(MAX_STEPS):
        if not pending:
            break
        # execute() returns the next task id, a list of them when an agent delegated to
        # several at once, or None when this branch is finished.
        result = Task.load(pending.pop(0)).execute(storage, client, sink, request_builder)
        if result is not None:
            pending.extend(result if isinstance(result, list) else [result])

    answers = sink.messages()
    return (answers[-1] if answers else None), sink, task.task_id


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "What is the weather in Kyoto?"
    answer, sink, task_id = run(question)

    print("\n" + "=" * 72)
    print(f"Q: {question}\n")
    for line in sink.progress():
        print(f"   [*] {line}")
    print(f"\nA: {answer}")
    print("\n" + render_trace(build_trace(task_id)))
    print("=" * 72)
