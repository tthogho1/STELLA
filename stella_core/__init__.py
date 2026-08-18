"""
STELLA's agent runtime as a standalone library, independent of the Flask server.

Task, Agent and AgentStorage drive the delegation loop; ChatQueue/TaskManager run it on
background threads; EventSink is where results and progress go; db exposes whichever
DatabaseInterface backend the DATABASE environment variable selects. app.server is
STELLA's own reference Flask application built on top of this package -- it is one
possible host, not the only one.

Deliberately empty otherwise: import the submodule you need (e.g.
`from stella_core.events import CollectingSink`) rather than through here, so that code
which only wants one piece -- say, just the EventSink hierarchy -- does not also import
the database layer or the OpenAI client as a side effect.

This package never loads a .env file itself: reading configuration is the host
application's job, and it must happen before importing anything below that reads an
environment variable at import time (most modules here do).
"""
