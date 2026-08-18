"""
Where the agent runtime sends things.

The runtime used to be handed a Flask-SocketIO instance directly. It only ever called
one method on it -- emit() -- but taking the transport itself meant the task loop could
not run without a web server, and every call site hand-rolled the same room/namespace
arguments and the same JSON envelope for progress.

An EventSink is that one method plus the two messages the runtime actually sends. The
server passes SocketIOSink; anything embedding the runtime as a library can pass
CollectingSink, or its own implementation.
"""
import json

# The SocketIO namespace the client connects to. Only the SocketIO sink cares.
CHAT_NAMESPACE = '/chat'

# Events the runtime produces. 'message' is the answer and ends the request as far as the
# client is concerned; 'chat_information' is progress and must not.
MESSAGE_EVENT = 'message'
INFORMATION_EVENT = 'chat_information'


class EventSink:
    """
    Base for anything the runtime can report to.

    Subclasses implement emit(). It keeps SocketIO's signature because an agent's
    on_completion callback is handed this object and existing agents call
    emit('message', text, room=chat_id, namespace='/chat') on it directly.
    """

    def emit(self, event, data=None, room=None, namespace=None):
        raise NotImplementedError

    def send_message(self, chat_id, text):
        """The answer to the user. Ends the request from the client's point of view."""
        self.emit(MESSAGE_EVENT, text, room=chat_id, namespace=CHAT_NAMESPACE)

    def send_progress(self, chat_id, message, kind="progress"):
        """
        A progress line. Deliberately a different event from send_message: the client
        treats a message as the end of the request, so progress arriving there would let
        the user type into a request that is still running.
        """
        self.emit(
            INFORMATION_EVENT,
            json.dumps({"type": kind, "message": message, "chat_id": chat_id}),
            room=chat_id,
            namespace=CHAT_NAMESPACE,
        )


class SocketIOSink(EventSink):
    """Sends to connected clients over Flask-SocketIO."""

    def __init__(self, socketio):
        self.socketio = socketio

    def emit(self, event, data=None, room=None, namespace=None):
        # flask_socketio rejects a positional None where a payload is expected.
        if data is None:
            self.socketio.emit(event, room=room, namespace=namespace)
        else:
            self.socketio.emit(event, data, room=room, namespace=namespace)


class CollectingSink(EventSink):
    """
    Keeps events in memory. For embedding the runtime without a server, and for tests.
    """

    def __init__(self):
        self.events = []

    def emit(self, event, data=None, room=None, namespace=None):
        self.events.append({"event": event, "data": data, "room": room})

    def messages(self):
        """Every answer sent, oldest first."""
        return [e["data"] for e in self.events if e["event"] == MESSAGE_EVENT]

    def progress(self):
        """Every progress line, as text."""
        out = []
        for e in self.events:
            if e["event"] != INFORMATION_EVENT:
                continue
            try:
                out.append(json.loads(e["data"]).get("message"))
            except (TypeError, ValueError):
                out.append(e["data"])
        return out


class NullSink(EventSink):
    """Discards everything. For a run whose output is read from the database instead."""

    def emit(self, event, data=None, room=None, namespace=None):
        pass
