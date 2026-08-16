"""
The seam between the agent runtime and whatever is listening.

The runtime used to be handed the Flask-SocketIO instance itself, which meant the task
loop could not run without a web server. It now takes an EventSink.
"""
import json

from app.events import (CollectingSink, EventSink, NullSink, SocketIOSink,
                        INFORMATION_EVENT, MESSAGE_EVENT)


class _Recorder:
    def __init__(self):
        self.calls = []

    def emit(self, event, *args, **kwargs):
        self.calls.append((event, args, kwargs))


def test_the_runtime_does_not_import_a_web_framework():
    """The whole point of the seam: importing the runtime must not pull in Flask."""
    import subprocess
    import sys

    check = (
        "import app.models.task, app.task_manager, app.chat_queue, app.events, sys;"
        "print(','.join(m for m in ('flask','flask_socketio','werkzeug') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True,
                         cwd="/Users/tthogho1/Documents/source/STELLA")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", f"runtime pulled in {out.stdout.strip()}"


class TestSink:
    def test_a_message_is_addressed_to_the_chat(self):
        sink = CollectingSink()
        sink.send_message("chat-1", "the answer")

        assert sink.events == [{"event": MESSAGE_EVENT, "data": "the answer", "room": "chat-1"}]
        assert sink.messages() == ["the answer"]

    def test_progress_goes_on_a_different_event_from_the_answer(self):
        """A chat message ends the request for the client; progress must not."""
        sink = CollectingSink()
        sink.send_progress("chat-1", "Asking WEATHER")

        assert sink.events[0]["event"] == INFORMATION_EVENT
        assert json.loads(sink.events[0]["data"]) == {
            "type": "progress", "message": "Asking WEATHER", "chat_id": "chat-1"}
        assert sink.progress() == ["Asking WEATHER"]
        assert sink.messages() == [], "progress must not read as an answer"

    def test_the_kind_can_be_overridden(self):
        sink = CollectingSink()
        sink.send_progress("chat-1", "still working", kind="busy")
        assert json.loads(sink.events[0]["data"])["type"] == "busy"

    def test_null_sink_discards_everything(self):
        sink = NullSink()
        sink.send_message("chat-1", "gone")
        sink.send_progress("chat-1", "gone")  # must not raise

    def test_the_base_class_demands_an_implementation(self):
        try:
            EventSink().send_message("chat-1", "x")
        except NotImplementedError:
            return
        raise AssertionError("EventSink.emit should have to be implemented")


class TestSocketIOSink:
    def test_it_forwards_room_and_namespace(self):
        recorder = _Recorder()
        SocketIOSink(recorder).send_message("chat-1", "hello")

        event, args, kwargs = recorder.calls[0]
        assert event == MESSAGE_EVENT
        assert args == ("hello",)
        assert kwargs == {"room": "chat-1", "namespace": "/chat"}

    def test_a_payload_less_emit_is_not_passed_as_none(self):
        """flask_socketio rejects a positional None where a payload is expected."""
        recorder = _Recorder()
        SocketIOSink(recorder).emit("ping", room="chat-1", namespace="/chat")

        _, args, _ = recorder.calls[0]
        assert args == ()


def test_agents_can_still_call_emit_directly():
    """on_completion is handed this object, and existing agents use SocketIO's API on it
    (see StellaCactusAgent.scroll_text)."""
    sink = CollectingSink()
    sink.emit("message", "a line", room="chat-1", namespace="/chat")

    assert sink.messages() == ["a line"]
