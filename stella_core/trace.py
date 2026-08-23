"""
Reconstructing what a request actually did.

Only the final answer reaches the user, and the progress events are gone as soon as they
are printed, so when a request behaves oddly -- an agent called four times for something
it already had, a minute spent somewhere -- there is nothing left to look at. Every task
is already in the database with its parent and the order it was delegated in; this reads
them back into the tree they formed and puts the timings next to it.

    from stella_core.trace import build_trace, render_trace
    print(render_trace(build_trace(top_level_task_id)))
"""
from stella_core.db import db

# Guard against a cycle in parent_task_id, which would otherwise recurse forever.
MAX_TREE_DEPTH = 100


def _sortable_id(task_id):
    """
    A tie-break key that works for either backend's ids.

    Numeric ids sort numerically -- "10" after "9", not before it -- and anything else
    falls back to the string. The tuple keeps the two kinds from being compared with each
    other, which would raise.
    """
    text = str(task_id)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def build_trace(top_level_task_id):
    """
    Reads every task of one request back into a tree.

    :param top_level_task_id: The id of the request's top level task
    :return: {"task_id", "agent", "runs", "seconds", "children": [...], ...} or None
    """
    tasks = {t["task_id"]: t for t in db.get_tasks_for_top_level(str(top_level_task_id))}
    if not tasks:
        return None

    children = {}
    for task in tasks.values():
        parent = task.get("parent_task_id")
        if parent is not None and str(parent) in tasks:
            children.setdefault(str(parent), []).append(task)

    # Siblings are shown in the order they were delegated, which is the order the parent
    # asked for them and the order it reads their results back. The task id only breaks a
    # tie, and it is not always a number: SQLite hands out integers, MongoDB hex
    # ObjectIds, and int() on one of those raised.
    def order(task):
        index = task.get("child_index")
        return (index if index is not None else 0, _sortable_id(task["task_id"]))

    for siblings in children.values():
        siblings.sort(key=order)

    def node(task, depth=0):
        runs = task.get("runs") or []
        return {
            "task_id": task["task_id"],
            "agent": task.get("current_agent"),
            "runs": runs,
            "run_count": len(runs),
            "seconds": round(sum(r.get("seconds") or 0 for r in runs), 3),
            "notes": [r["note"] for r in runs if r.get("note")],
            "errors": [r["error"] for r in runs if r.get("error")],
            "memories": len(task.get("memories") or []),
            "children": ([] if depth >= MAX_TREE_DEPTH
                         else [node(c, depth + 1) for c in children.get(task["task_id"], [])]),
        }

    root = tasks.get(str(top_level_task_id)) or next(iter(tasks.values()))
    tree = node(root)
    tree["totals"] = _totals(tree)
    return tree


def _totals(tree):
    tasks = agent_runs = 0
    wall_start, wall_end = None, None

    def walk(n):
        nonlocal tasks, agent_runs, wall_start, wall_end
        tasks += 1
        agent_runs += n["run_count"]
        for r in n["runs"]:
            if r.get("started") is not None:
                wall_start = r["started"] if wall_start is None else min(wall_start, r["started"])
            if r.get("ended") is not None:
                wall_end = r["ended"] if wall_end is None else max(wall_end, r["ended"])
        for c in n["children"]:
            walk(c)

    walk(tree)
    return {
        "tasks": tasks,
        "agent_runs": agent_runs,
        # Wall clock, not the sum of the spans: siblings overlap when they run in parallel.
        # `is not None`, not a truth test: a run starting at timestamp 0 is falsy and
        # would silently report the whole request as taking no time at all.
        "wall_seconds": (round(wall_end - wall_start, 3)
                         if wall_start is not None and wall_end is not None else 0.0),
    }


def render_trace(tree, show_task_ids=False):
    """Renders a trace as an indented tree. Returns "" for an empty trace."""
    if not tree:
        return ""

    lines = []

    def label(n):
        parts = [n["agent"] or "?"]
        if show_task_ids:
            parts.append(f"#{n['task_id']}")
        parts.append(f"{n['seconds']:.1f}s")
        if n["run_count"] > 1:
            parts.append(f"x{n['run_count']}")
        text = " ".join(parts)
        if n["notes"]:
            text += f"  — {n['notes'][-1]}"
        if n["errors"]:
            text += f"  !! {n['errors'][-1]}"
        return text

    def walk(n, prefix="", last=True, root=False):
        if root:
            lines.append(label(n))
        else:
            lines.append(f"{prefix}{'└─ ' if last else '├─ '}{label(n)}")
            prefix += "   " if last else "│  "
        for i, child in enumerate(n["children"]):
            walk(child, prefix, i == len(n["children"]) - 1)

    walk(tree, root=True)

    t = tree.get("totals") or {}
    lines.append("")
    lines.append(f"{t.get('tasks', 0)} tasks, {t.get('agent_runs', 0)} agent runs, "
                 f"{t.get('wall_seconds', 0)}s wall clock")
    return "\n".join(lines)
