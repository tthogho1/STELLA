"""
When an agent delegates to several agents at once, the children run as siblings and the
parent must wake up exactly once, holding every result, in the order it asked for them.

Both halves of that are races. Two siblings finishing together must not both decide they
were last, and two siblings storing results must not lose each other's work. Ordering is
a correctness concern rather than cosmetics: a short result landing behind a large JSON
payload is reliably overlooked by the action-selection model.
"""
import random
import threading
import time


def _release_together(fn, n):
    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait()
        fn(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


class TestJoinCounter:
    def test_exactly_one_sibling_is_last(self, db, make_task):
        for siblings in (2, 5, 25):
            task_id = make_task(pending_children=siblings)
            winners = []

            def finish(_):
                if db.decrement_pending_children(task_id) == 0:
                    winners.append(True)

            _release_together(finish, siblings)

            assert len(winners) == 1, f"{siblings} siblings produced {len(winners)} winners"
            assert db.get_task_data(task_id)["pending_children"] == 0

    def test_counter_never_goes_negative(self, db, make_task):
        task_id = make_task(pending_children=1)
        assert [db.decrement_pending_children(task_id) for _ in range(4)] == [0, 0, 0, 0]


class TestResultOrdering:
    def test_results_follow_delegation_order_not_completion_order(self, db, make_task):
        """Children write into the slot they were delegated in, so a child that finishes
        first cannot push an earlier sibling's result behind its own."""
        from app.models.task import Task

        for completion_order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1]):
            task_id = make_task(memories=["seed"], pending_children=4)

            def finish(position):
                child = completion_order[position]
                time.sleep(0.01 * position)
                db.store_child_result(task_id, child, [f"result-{child}"])

            _release_together(finish, 4)

            parent = Task.load(task_id)
            parent._collect_child_results()
            assert parent.memories == ["seed"] + [f"result-{i}" for i in range(4)], \
                f"completion order {completion_order} changed the result order"

    def test_concurrent_slot_writes_keep_every_result(self, db, make_task):
        task_id = make_task(pending_children=30)
        _release_together(lambda i: db.store_child_result(task_id, i, [f"r{i}"]), 30)

        slots = db.get_task_data(task_id)["pending_results"]
        collected = [m for key in sorted(slots, key=int) for m in slots[key]]
        assert collected == [f"r{i}" for i in range(30)]

    def test_random_completion_orders(self, db, make_task):
        from app.models.task import Task

        for _ in range(10):
            order = list(range(4))
            random.shuffle(order)
            task_id = make_task(pending_children=4)
            for position, child in enumerate(order):
                db.store_child_result(task_id, child, [f"result-{child}"])
            parent = Task.load(task_id)
            parent._collect_child_results()
            assert parent.memories == [f"result-{i}" for i in range(4)]

    def test_a_child_that_gave_up_leaves_a_gap(self, db, make_task):
        """A subtask stopped by the depth limit never stores anything."""
        from app.models.task import Task

        task_id = make_task()
        db.store_child_result(task_id, 0, ["first"])
        db.store_child_result(task_id, 2, ["third"])

        parent = Task.load(task_id)
        parent._collect_child_results()
        assert parent.memories == ["first", "third"]

    def test_collecting_twice_does_not_duplicate(self, db, make_task):
        from app.models.task import Task

        task_id = make_task()
        db.store_child_result(task_id, 0, ["only"])

        Task.load(task_id)._collect_child_results()
        again = Task.load(task_id)
        again._collect_child_results()
        assert again.memories == ["only"]


class _ForwardingAgent:
    """Stands in for an Agent, carrying only the flags forwarding looks at."""

    def __init__(self, all_entries=False, last_only=False):
        self.forward_all_memory_entries_to_parent = all_entries
        self.forward_last_memory_to_parent = last_only


class TestMemoryInheritance:
    def test_a_child_forwards_only_what_it_produced(self, db, make_task):
        """A child is created holding the parent's memories as context. Forwarding those
        back doubles the parent's memories every round, which grew the prompt
        exponentially until it overflowed the context window."""
        from app.models.task import Task

        parent_id = make_task(memories=["A", "B"], pending_children=1)
        child_id = make_task(memories=["A", "B"], parent_task_id=parent_id,
                             inherited_memory_count=2, child_index=0)

        child = Task.load(child_id)
        child.memories.append("C")
        child.forward_memories_to_parent(_ForwardingAgent(all_entries=True))

        parent = Task.load(parent_id)
        parent._collect_child_results()
        assert parent.memories == ["A", "B", "C"], \
            "the child returned entries the parent already had"

    def test_forwarding_stays_linear_over_several_rounds(self, db, make_task):
        """The failure this guards against only shows up once it compounds."""
        from app.models.task import Task

        parent_id = make_task()
        for round_number in range(6):
            parent = Task.load(parent_id)
            child_id = make_task(memories=list(parent.memories), parent_task_id=parent_id,
                                 inherited_memory_count=len(parent.memories), child_index=0)
            child = Task.load(child_id)
            child.memories.append(f"result-{round_number}")
            child.forward_memories_to_parent(_ForwardingAgent(all_entries=True))
            Task.load(parent_id)._collect_child_results()

        final = Task.load(parent_id).memories
        assert final == [f"result-{i}" for i in range(6)], \
            f"expected one entry per round, got {len(final)}"

    def test_forward_last_memory_sends_one_entry(self, db, make_task):
        from app.models.task import Task

        parent_id = make_task(memories=["A"])
        child_id = make_task(memories=["A"], parent_task_id=parent_id,
                             inherited_memory_count=1, child_index=0)

        child = Task.load(child_id)
        child.memories += ["B", "C"]
        child.forward_memories_to_parent(_ForwardingAgent(last_only=True))

        parent = Task.load(parent_id)
        parent._collect_child_results()
        assert parent.memories == ["A", "C"]

    def test_an_agent_with_no_forwarding_flags_sends_nothing(self, db, make_task):
        from app.models.task import Task

        parent_id = make_task()
        child_id = make_task(parent_task_id=parent_id, child_index=0)

        child = Task.load(child_id)
        child.memories.append("private")
        child.forward_memories_to_parent(_ForwardingAgent())

        assert db.get_task_data(parent_id)["pending_results"] == {}
