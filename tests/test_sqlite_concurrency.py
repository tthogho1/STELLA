"""
The SQLite backend shares one connection across every ChatQueue worker, TaskManager
worker and request handler. sqlite3 forbids that, and the server only survives it in
production because gevent turns those workers into greenlets on a single OS thread.
These tests use real threads, which is where the original code raised
"SQLite objects created in a thread can only be used in that same thread" on every call.
"""
import json
import sqlite3
import threading


def _run_concurrently(fn, n):
    """Runs fn(i) on n real threads, released together, and returns any exceptions."""
    errors = []
    barrier = threading.Barrier(n)

    def worker(i):
        try:
            barrier.wait()
            fn(i)
        except Exception as e:  # noqa: BLE001 - the point is to surface anything raised
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_usable_from_many_real_threads(sqlite_db):
    def work(i):
        user = sqlite_db.create_user(f"user{i}", b"pw")
        workspace = sqlite_db.create_workspace(user.id, f"ws{i}", {})
        sqlite_db.get_workspace(workspace.id)
        chat = sqlite_db.create_chat(workspace.id, user.id)
        sqlite_db.get_chat_by_id(chat.chat_id)

    assert _run_concurrently(work, 20) == []


def test_no_writes_are_lost(sqlite_db):
    _run_concurrently(lambda i: sqlite_db.create_user(f"u{i}", b"pw"), 20)
    count = sqlite_db.conn.execute("SELECT COUNT(*) FROM user").fetchone()[0]
    assert count == 20


def test_read_modify_write_is_not_clobbered(sqlite_db):
    """create_workspace reads a user's workspace list, appends and writes it back."""
    user = sqlite_db.create_user("solo", b"pw")
    _run_concurrently(lambda i: sqlite_db.create_workspace(user.id, f"ws{i}", {}), 15)

    row = sqlite_db.conn.execute("SELECT workspaces FROM user WHERE id = ?", (user.id,)).fetchone()
    assert len(json.loads(row["workspaces"])) == 15


def test_reentrant_lock_does_not_deadlock(sqlite_db):
    """get_user_workspaces() calls get_workspace(); both hold the same lock."""
    user = sqlite_db.create_user("nested", b"pw")
    sqlite_db.create_workspace(user.id, "ws", {})
    assert len(sqlite_db.get_user_workspaces(user.id)) == 1


def test_wal_is_enabled(sqlite_db):
    assert sqlite_db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_migrates_a_database_created_before_the_new_columns(tmp_path):
    """CREATE TABLE IF NOT EXISTS leaves an existing table alone, so columns added later
    have to be applied with ALTER TABLE or an upgraded install cannot read its own data."""
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.execute(
        'CREATE TABLE "task" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "chat_id" INTEGER,'
        ' "agents" TEXT, "owner" INTEGER, "coordinator_agent" TEXT, "current_agent" TEXT,'
        ' "memories" TEXT, "parent_task_id" INTEGER, "top_level_task_id" INTEGER,'
        ' "completed" INTEGER DEFAULT 0, "created_at" TEXT, "depths" TEXT,'
        ' "is_top_level" INTEGER DEFAULT 0, "top_level_task_max_depth" INTEGER,'
        ' "top_level_task_depth" INTEGER)'
    )
    legacy.execute(
        "INSERT INTO task (chat_id, agents, owner, coordinator_agent, current_agent, memories)"
        " VALUES ('1', '{}', '1', 'c', 'c', '[\"legacy row\"]')"
    )
    legacy.commit()
    legacy.close()

    import stella_core.db.databases.sqlite as sqlite_module
    original = sqlite_module.SQLITE_DB_PATH
    sqlite_module.SQLITE_DB_PATH = str(path)
    try:
        upgraded = sqlite_module.SQLite()
        task = upgraded.get_task_data("1")
    finally:
        sqlite_module.SQLITE_DB_PATH = original

    assert task["memories"] == ["legacy row"]
    assert task["pending_children"] == 0
    assert task["inherited_memory_count"] == 0
    assert task["pending_results"] == {}
