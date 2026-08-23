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
import uuid

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


def _mongo_uri():
    """The MongoDB to test against. Point STELLA_TEST_MONGO_URI elsewhere to override."""
    return os.environ.get("STELLA_TEST_MONGO_URI", "mongodb://localhost:27017")


def _mongo_available(uri):
    try:
        import pymongo
    except ImportError:
        return False
    try:
        pymongo.MongoClient(uri, serverSelectionTimeoutMS=1500).server_info()
        return True
    except Exception:
        return False


@pytest.fixture(params=["sqlite", "mongodb"])
def db(request, monkeypatch):
    """
    A throwaway backend, once per implementation.

    Every DB-backed test runs against both, because the two are not the same code: the
    atomicity the parallel join depends on is a lock plus a read-modify-write in SQLite
    and $inc/$set in MongoDB, and only the first was ever exercised. MongoDB is skipped
    when there is no server to talk to, so the suite still runs anywhere.

    stella_core.db.db is a proxy forwarding to whichever backend is current, so patching
    the module's `_current` is enough for every consumer -- Task, TaskManager, ChatQueue,
    the views -- to see the swap.
    """
    import stella_core.db

    if request.param == "sqlite":
        import stella_core.db.databases.sqlite as sqlite_module

        path = os.path.join(tempfile.mkdtemp(prefix="stella-case-"), "case.db")
        monkeypatch.setenv("SQLITE_DB_PATH", path)
        monkeypatch.setattr(sqlite_module, "SQLITE_DB_PATH", path)
        backend = sqlite_module.SQLite()
    else:
        uri = _mongo_uri()
        if not _mongo_available(uri):
            pytest.skip(f"no MongoDB at {uri}")

        import pymongo
        import stella_core.db.databases.mongodb as mongo_module

        # A database of its own per test, dropped after, so cases cannot collide.
        name = f"stella_test_{uuid.uuid4().hex[:12]}"
        monkeypatch.setattr(mongo_module, "MONGO_URI", uri)
        monkeypatch.setattr(mongo_module, "MONGO_DB_NAME", name)
        backend = mongo_module.MongoDB()
        request.addfinalizer(lambda: pymongo.MongoClient(uri).drop_database(name))

    monkeypatch.setattr(stella_core.db, "_current", backend)

    return backend


@pytest.fixture
def sqlite_db(monkeypatch):
    """
    A SQLite backend specifically.

    tests/test_sqlite_concurrency.py is about this backend's own hazards -- one shared
    connection, the ALTER TABLE migrations -- and reaches into db.conn, so it cannot use
    the parametrized fixture.
    """
    import stella_core.db
    import stella_core.db.databases.sqlite as sqlite_module

    path = os.path.join(tempfile.mkdtemp(prefix="stella-sqlite-"), "case.db")
    monkeypatch.setenv("SQLITE_DB_PATH", path)
    monkeypatch.setattr(sqlite_module, "SQLITE_DB_PATH", path)
    backend = sqlite_module.SQLite()
    monkeypatch.setattr(stella_core.db, "_current", backend)
    return backend


@pytest.fixture
def missing_id(db):
    """An id of the right shape for this backend that nothing was ever stored under."""
    import uuid as _uuid
    return "999999" if type(db).__name__ == "SQLite" else _uuid.uuid4().hex[:24].rjust(24, "0")


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
