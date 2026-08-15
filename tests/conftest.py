"""
Shared setup for the test suite.

Two things have to happen before anything under app/ is imported:

- SQLITE_DB_PATH is read at module scope by app/db/databases/sqlite.py, and app/db builds
  a singleton connection at import time. Point it at a throwaway file so a test run never
  touches app/sqlite.db.
- app/openai_client.py constructs the OpenAI client at import time and the SDK refuses to
  start without credentials. Nothing here calls the API; the key only has to exist.

Nothing in this suite makes a network request.
"""
import os
import tempfile

import pytest

_TMP = tempfile.mkdtemp(prefix="stella-tests-")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-a-real-key")
os.environ["SQLITE_DB_PATH"] = os.path.join(_TMP, "test.db")
# Pin the prompt caps so the assertions do not depend on a developer's app/.env.
os.environ["MAX_MEMORY_ENTRIES"] = "20"
os.environ["MAX_MEMORY_ENTRY_CHARS"] = "8000"
os.environ["MAX_CHAT_HISTORY_MESSAGES"] = "20"


@pytest.fixture
def db(monkeypatch):
    """
    A SQLite backend on its own file, so tests cannot interfere with each other.

    app/db builds a module-level singleton at import time and Task binds it with
    `from app.db import db`, so both names have to be redirected or Task.load() would
    silently read a different database than the fixture wrote to.
    """
    import app.db
    import app.db.databases.sqlite as sqlite_module

    path = os.path.join(tempfile.mkdtemp(prefix="stella-case-"), "case.db")
    monkeypatch.setenv("SQLITE_DB_PATH", path)
    monkeypatch.setattr(sqlite_module, "SQLITE_DB_PATH", path)

    backend = sqlite_module.SQLite()
    monkeypatch.setattr(app.db, "db", backend)

    import app.models.task
    monkeypatch.setattr(app.models.task, "db", backend)

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
    from app.models.chat import Chat

    chat = Chat.__new__(Chat)
    chat.chat_history = [{"role": "user", "content": m} for m in messages]
    return chat
