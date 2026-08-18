"""Compatibility re-export: the real implementation moved to stella_core.events."""
from stella_core.events import (
    EventSink, SocketIOSink, CollectingSink, NullSink,
    CHAT_NAMESPACE, MESSAGE_EVENT, INFORMATION_EVENT,
)
