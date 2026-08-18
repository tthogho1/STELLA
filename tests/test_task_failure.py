"""
What happens to the rest of the tree when one task raises.

A subtask that dies still owes its parent an answer. The parent is waiting on a fixed
number of children, so a child that disappears without releasing its slot parks the
parent forever -- and takes a successful sibling's result down with it.
"""
import threading
import types

from stella_core.events import CollectingSink


def _worker(db, events=None):
    """A Worker with its queue and pool stubbed out."""
    from stella_core.task_manager import Worker
    import stella_core.task_manager

    worker = Worker.__new__(Worker)
    # Worker subclasses Thread, and Thread.name refuses to be set before __init__ runs.
    threading.Thread.__init__(worker, name="test-worker")
    queued = []
    worker.manager = types.SimpleNamespace(
        events=events or CollectingSink(),
        add_task=queued.append,
    )
    worker.queued = queued
    stella_core.task_manager.db = db
    return worker


def test_a_failed_child_releases_its_parents_slot(db, make_task):
    parent_id = make_task(pending_children=2)
    child_id = make_task(parent_task_id=parent_id, child_index=0)

    worker = _worker(db)
    worker._handle_failure(child_id)

    assert db.get_task_data(parent_id)["pending_children"] == 1
    assert worker.queued == [], "the parent still has a sibling outstanding"


def test_the_parent_runs_again_once_the_last_child_fails(db, make_task):
    """The surviving sibling's result must still reach the user."""
    parent_id = make_task(pending_children=2)
    good_id = make_task(parent_task_id=parent_id, child_index=0)
    bad_id = make_task(parent_task_id=parent_id, child_index=1)

    db.store_child_result(parent_id, 0, ["the good result"])
    db.decrement_pending_children(parent_id)          # the healthy child reports in

    worker = _worker(db)
    worker._handle_failure(bad_id)

    assert db.get_task_data(parent_id)["pending_children"] == 0
    assert worker.queued == [parent_id], "the parent should be re-queued"
    assert db.get_task_data(parent_id)["pending_results"] == {"0": ["the good result"]}


def test_the_user_is_told_which_agent_failed(db, make_task):
    parent_id = make_task(pending_children=1)
    child_id = make_task(parent_task_id=parent_id, child_index=0)

    sink = CollectingSink()
    _worker(db, sink)._handle_failure(child_id)

    assert any("failed" in (p or "") for p in sink.progress())
    assert sink.messages() == [], "a failed subtask must not look like the answer"


def test_a_failed_top_level_task_ends_the_request(db, make_task):
    """Nothing above it to report to, so the chat has to be unblocked instead."""
    user = db.create_user("u", b"pw")
    workspace = db.create_workspace(user.id, "ws", {})
    chat = db.create_chat(workspace.id, user.id)
    chat.busy = True
    db.update_chat(chat)

    task = db.create_task(chat_id=chat.chat_id, agents={}, owner=user.id,
                          coordinator_agent="c", current_agent="c", memories=[])

    sink = CollectingSink()
    worker = _worker(db, sink)
    worker._handle_failure(task["task_id"])

    assert db.get_chat_by_id(chat.chat_id).busy is False
    assert worker.queued == []
    assert len(sink.messages()) == 1, "the user has to be told the request ended"


def test_an_unloadable_task_still_unblocks_the_chat(db):
    worker = _worker(db)
    worker._handle_failure("does-not-exist")          # must not raise
    assert worker.queued == []
