"""
Reconstructing what a request did.

Only the final answer reaches the user and progress events are gone once printed, so the
trace is the only way to see afterwards that an agent ran five times for something it
already had. The tree is rebuilt from parent_task_id, which means the reconstruction has
to survive the shapes real runs produce: a task that ran several times, siblings that
finished out of order, a branch that gave up.
"""
import pytest

from stella_core.trace import build_trace, render_trace


def _run(started, seconds, note=None, error=None):
    return {"agent": "a", "started": started, "ended": started + seconds,
            "seconds": seconds, "note": note, "error": error}


@pytest.fixture
def tree(db):
    """A three-level tree: coordinator -> planner -> weather, plus a sibling."""
    def _make(agent, parent=None, top=None, child_index=None, runs=None, is_top=False):
        data = db.create_task(chat_id="1", agents={}, owner="1", coordinator_agent="c",
                              current_agent=agent, memories=[], parent_task_id=parent,
                              top_level_task_id=top, child_index=child_index,
                              is_top_level=is_top, runs=runs or [])
        return data["task_id"]

    root = _make("coordinator", is_top=True, runs=[_run(100, 1.0), _run(110, 0.5, "answered the user")])
    db.update_task_data({**db.get_task_data(root), "top_level_task_id": root})

    planner = _make("planner", parent=root, top=root, child_index=0,
                    runs=[_run(101, 2.0, "delegated to WEATHER")])
    _make("weather", parent=planner, top=root, child_index=0, runs=[_run(102, 1.5)])
    _make("brewery", parent=root, top=root, child_index=1, runs=[_run(101, 3.0)])
    return root


class TestShape:
    def test_the_tree_is_rebuilt_from_parent_ids(self, db, tree):
        t = build_trace(tree)

        assert t["agent"] == "coordinator"
        assert [c["agent"] for c in t["children"]] == ["planner", "brewery"]
        assert [g["agent"] for g in t["children"][0]["children"]] == ["weather"]

    def test_siblings_are_in_delegation_order(self, db):
        """Not the order they were created or finished -- the order the parent asked."""
        root = db.create_task(chat_id="1", agents={}, owner="1", coordinator_agent="c",
                              current_agent="root", memories=[], is_top_level=True)["task_id"]
        db.update_task_data({**db.get_task_data(root), "top_level_task_id": root})
        for agent, index in (("second", 1), ("first", 0), ("third", 2)):
            db.create_task(chat_id="1", agents={}, owner="1", coordinator_agent="c",
                           current_agent=agent, memories=[], parent_task_id=root,
                           top_level_task_id=root, child_index=index)

        t = build_trace(root)

        assert [c["agent"] for c in t["children"]] == ["first", "second", "third"]

    def test_an_unknown_task_has_no_trace(self, db):
        assert build_trace("999") is None


class TestTiming:
    def test_a_task_that_ran_twice_is_counted_once_with_both_spans(self, db, tree):
        """A parent runs again every time its children report back."""
        t = build_trace(tree)

        assert t["run_count"] == 2
        assert t["seconds"] == 1.5

    def test_wall_clock_spans_the_whole_request(self, db, tree):
        """Earliest start to latest end, so the gaps where a parent sat parked waiting for
        its children are counted too -- that idle time is part of what the user waited."""
        totals = build_trace(tree)["totals"]

        assert totals["tasks"] == 4
        assert totals["agent_runs"] == 5
        assert totals["wall_seconds"] == 10.5          # earliest start 100, latest end 110.5
        assert totals["wall_seconds"] > sum([1.0, 0.5, 2.0, 1.5, 3.0])

    def test_parallel_siblings_do_not_have_their_time_added_up(self, db):
        """Two agents running at once take as long as the slower one, not both combined."""
        root = db.create_task(chat_id="1", agents={}, owner="1", coordinator_agent="c",
                              current_agent="root", memories=[], is_top_level=True,
                              runs=[_run(0, 0.1)])["task_id"]
        db.update_task_data({**db.get_task_data(root), "top_level_task_id": root})
        for agent, index in (("a", 0), ("b", 1)):
            db.create_task(chat_id="1", agents={}, owner="1", coordinator_agent="c",
                           current_agent=agent, memories=[], parent_task_id=root,
                           top_level_task_id=root, child_index=index,
                           runs=[_run(0.1, 5.0)])          # both start at the same moment

        totals = build_trace(root)["totals"]

        assert totals["wall_seconds"] == 5.1
        assert totals["wall_seconds"] < 0.1 + 5.0 + 5.0

    def test_notes_and_errors_are_carried_through(self, db):
        root = db.create_task(chat_id="1", agents={}, owner="1", coordinator_agent="c",
                              current_agent="root", memories=[], is_top_level=True,
                              runs=[_run(1, 1, note="gave up: limit"),
                                    _run(3, 1, error="RuntimeError: boom")])["task_id"]
        db.update_task_data({**db.get_task_data(root), "top_level_task_id": root})

        t = build_trace(root)

        assert t["notes"] == ["gave up: limit"]
        assert t["errors"] == ["RuntimeError: boom"]


class TestRendering:
    def test_it_draws_the_tree(self, db, tree):
        out = render_trace(build_trace(tree))

        assert out.splitlines()[0].startswith("coordinator")
        assert "├─ planner" in out
        assert "└─ brewery" in out
        assert "weather" in out

    def test_a_repeated_task_is_marked(self, db, tree):
        out = render_trace(build_trace(tree))

        assert "x2" in out, "a task that ran twice has to be visible as such"
        assert "answered the user" in out

    def test_the_totals_line_is_there(self, db, tree):
        assert "4 tasks, 5 agent runs" in render_trace(build_trace(tree))

    def test_task_ids_are_optional(self, db, tree):
        assert "#" not in render_trace(build_trace(tree))
        assert "#" in render_trace(build_trace(tree), show_task_ids=True)

    def test_an_empty_trace_renders_empty(self):
        assert render_trace(None) == ""
