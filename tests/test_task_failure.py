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


class TestGiveUpIsReported:
    """
    A subtask stopped by a depth limit has to say so in the parent's memories.

    Releasing the join slot alone is not enough: the parent sees an empty result, reads it
    as "still nothing on this" and delegates again. Once a limit is reached every retry is
    refused on arrival, so the request spins until the overall depth cap ends it with no
    answer -- which is what a real coffee-search request did, ten calls and nothing back.
    """

    def _task(self, db, make_task, **kwargs):
        from stella_core.models.task import Task
        return Task.load(make_task(**kwargs))

    def test_the_reason_lands_in_the_parents_memories(self, db, make_task):
        from stella_core.events import CollectingSink

        parent_id = make_task(pending_children=1)
        child = self._task(db, make_task, parent_task_id=parent_id, child_index=0)

        child._report_to_parent(chat=None, events=CollectingSink(),
                                give_up_message="Reached the call limit for agent X (5/5).")

        stored = db.get_task_data(parent_id)["pending_results"]
        assert stored, "the parent got nothing back, so it will just delegate again"
        text = " ".join(stored["0"])
        assert "call limit" in text
        assert "not call this agent again" in text

    def test_it_goes_into_the_slot_it_was_delegated_in(self, db, make_task):
        from stella_core.events import CollectingSink

        parent_id = make_task(pending_children=2)
        child = self._task(db, make_task, parent_task_id=parent_id, child_index=1)

        child._report_to_parent(chat=None, events=CollectingSink(), give_up_message="stopped")

        assert list(db.get_task_data(parent_id)["pending_results"]) == ["1"]

    def test_the_slot_is_still_released(self, db, make_task):
        """The original reason this path exists must keep working."""
        from stella_core.events import CollectingSink

        parent_id = make_task(pending_children=1)
        child = self._task(db, make_task, parent_task_id=parent_id, child_index=0)

        requeue = child._report_to_parent(chat=None, events=CollectingSink(),
                                          give_up_message="stopped")

        assert db.get_task_data(parent_id)["pending_children"] == 0
        assert requeue == parent_id, "the last sibling has to wake the parent"

    def test_a_normal_completion_stores_nothing_extra(self, db, make_task):
        from stella_core.events import CollectingSink

        parent_id = make_task(pending_children=1)
        child = self._task(db, make_task, parent_task_id=parent_id, child_index=0)

        child._report_to_parent(chat=None, events=CollectingSink())

        assert db.get_task_data(parent_id)["pending_results"] == {}
