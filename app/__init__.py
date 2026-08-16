"""
STELLA.

Deliberately empty. Everything under app.* is imported through this module, so anything
placed here is loaded even by code that only wants the agent runtime -- create_app used
to live here, which meant `import app.models.task` pulled in Flask, SocketIO and every
view. The web server is app.server; the runtime (Task, Agent, AgentStorage,
OpenAIClient) has no web dependency and can be driven on its own.
"""
