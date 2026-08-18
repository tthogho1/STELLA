"""
Shared setup for the test suite.

Two things are set before anything under stella_core is imported:

- SQLITE_DB_PATH, read by stella_core/db/databases/sqlite.py whenever a SQLite backend is
  actually constructed (stella_core.db no longer does this automatically at import time --
  see the `db` fixture below). Point it at a throwaway file so a test run never touches
  app/sqlite.db.
- stella_core/openai_client.py constructs the OpenAI client at import time and the SDK
  refuses to start without credentials. Nothing here calls the API; the key only has to
  exist.

Nothing in this suite makes a network request.
"""
import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="stella-tests-")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
# stella_core.db reads this at import time to pick a backend, and (unlike app/db before
# it) stella_core never loads a .env file itself -- set it explicitly so a test run does
# not depend on a developer's local app/.env or shell environment already having it.
os.environ.setdefault("DATABASE", "sqlite")
os.environ["SQLITE_DB_PATH"] = os.path.join(_TMP, "test.db")
# Pin the prompt caps so the assertions do not depend on a developer's app/.env.
os.environ["MAX_MEMORY_ENTRIES"] = "20"
os.environ["MAX_MEMORY_ENTRY_CHARS"] = "8000"
os.environ["MAX_CHAT_HISTORY_MESSAGES"] = "20"


@pytest.fixture
def db(monkeypatch):
    """
    A SQLite backend on its own file, so tests cannot interfere with each other.

    stella_core.db.db is a proxy that forwards to whichever backend is currently set
    (normally by calling init_database(), which this fixture bypasses to build its own
    throwaway SQLite directly). Every consumer -- Task, TaskManager, ChatQueue, the Flask
    views -- holds the same proxy object, so patching the module's `_current` here is
    enough for all of them to see the swap; nothing else needs to be patched separately.
    """
    import stella_core.db
    import stella_core.db.databases.sqlite as sqlite_module

    path = os.path.join(tempfile.mkdtemp(prefix="stella-case-"), "case.db")
    monkeypatch.setenv("SQLITE_DB_PATH", path)
    monkeypatch.setattr(sqlite_module, "SQLITE_DB_PATH", path)

    backend = sqlite_module.SQLite()
    monkeypatch.setattr(stella_core.db, "_current", backend)

    return backend


@pytest.fixture
def make_task(db):
    """Creates a task row and returns its id."""
    def _make(memories=None, pending_children=0, parent_task_id=None,
              inherited_memory_count=0, child_index=None):
        data = db.create_task(
            chat_id="1", agents={}, owner="1", coordinator_agent="c", current_agent="c",
            memories=list(memories or []), pending_children=pending_children,
            parent_task_id=parent_task_id, inherited_memory_count=inherited_memory_count,
            child_index=child_index,
        )
        return data["task_id"]

    return _make


def make_chat(messages):
    """A Chat carrying the given history, without touching the database."""
    from stella_core.models.chat import Chat

    chat = Chat.__new__(Chat)
    chat.chat_history = [{"role": "user", "content": m} for m in messages]
    return chat
