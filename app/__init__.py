"""
STELLA.

Deliberately empty. The agent runtime (Task, Agent, AgentStorage, EventSink, ChatQueue,
TaskManager, the db layer, OpenAIClient) lives in stella_core and has no web dependency;
app is STELLA's reference Flask server built on top of it, plus the built-in, demo and
custom agents that ship with it. app.models, app.db, app.agent_storage, app.chat_queue,
app.task_manager, app.openai_client, app.utils and app.events are now thin re-exports of
the same objects from stella_core, kept so existing agents and integrations that import
from app.* keep working unchanged.
"""
