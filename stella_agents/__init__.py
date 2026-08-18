"""
The agent content that ships with STELLA's reference server.

A plain, ordinary Python package -- nothing here is special to stella_core or app.
AgentStorage discovers Agent subclasses by scanning directories it is handed
(AgentStorage(agent_dirs=[...])), not by importing this package, so a different "用途別"
(purpose-specific) deployment can point it at a completely different package instead of,
or alongside, this one. app/server.py is the one that decides to use stella_agents.
"""
