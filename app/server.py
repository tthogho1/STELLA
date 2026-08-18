import os

from dotenv import load_dotenv

# Loaded before importing stella_core: several of its modules read configuration from
# the environment at import time, and stella_core never loads a .env file itself -- that
# is the host application's job (see stella_core/__init__.py).
load_dotenv()

from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO

import stella_agents
from stella_core.agent_storage import AgentStorage
from stella_core.events import SocketIOSink
from stella_core.chat_queue import ChatQueue
from stella_core.db import init_database
from stella_core.models.task import configure_default_agents
from app.config import flask_configs

from app.views.auth import auth_views
from app.views.chat import initiate_chat_views
from app.views.workspace import workspace_views
from app.views.agent import agent_views
from app.views.utils import utils_views
from app.views.user import user_views

# The agent content that ships with STELLA's reference server. Resolved through the
# stella_agents package rather than a path relative to this file, since -- unlike
# DOWNLOADED_AGENTS_DIR below -- it does not have to live next to app/ on disk.
STELLA_AGENTS_DIR = os.path.dirname(os.path.abspath(stella_agents.__file__))
# Where GET /agent/download (app/views/agent.py) installs community packages at runtime.
# Kept as a second, separate scan directory rather than mixed into stella_agents/, so
# runtime-installed content never overwrites anything this repo ships.
DOWNLOADED_AGENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agents')


def create_app(host, port):
    """
    Creates the Flask app and registers the blueprints.
    Loads the configuration from the config.py file.
    :return:
    """
    config_name = os.getenv('FLASK_CONFIG', 'default')

    # Picks the backend from DATABASE (see stella_core/db/__init__.py). Explicit and
    # first, since the views imported above already hold a `db` reference that only
    # becomes usable once this has run.
    init_database()

    # stella_core has no built-in notion of a coordinator or welcome agent -- these are
    # STELLA's own defaults, from stella_agents/StellaAgents/ (see
    # stella_core/models/task.py:configure_default_agents).
    configure_default_agents(
        general_agent_id='stella_coordinator_agent',
        empty_workspace_agent_id='stella_welcome_agent',
    )

    socketio = SocketIO(ping_timeout=900, ping_interval=60)
    app = Flask(__name__)
    CORS(app, origins="*", supports_credentials=True)
    JWTManager(app)

    socketio.init_app(
        app,
        cors_allowed_origins="*",
        async_mode=flask_configs[config_name].ASYNC_MODE,
        logger=flask_configs[config_name].SOCKET_LOGGER,
        engineio_logger=flask_configs[config_name].ENGINEIO_LOGGER
    )

    app.config.from_object(flask_configs[config_name])
    app.extensions['socketio'] = socketio
    app.extensions['agent_storage'] = AgentStorage(agent_dirs=[STELLA_AGENTS_DIR, DOWNLOADED_AGENTS_DIR])
    # The runtime is handed an EventSink rather than the SocketIO instance itself, so the
    # task loop can also be driven without a web server (see app/events.py).
    app.extensions['event_sink'] = SocketIOSink(socketio)
    app.extensions['chat_queue'] = ChatQueue(num_workers=5, events=app.extensions['event_sink'], agent_storage=app.extensions['agent_storage'])

    app.register_blueprint(auth_views)
    app.register_blueprint(user_views)
    app.register_blueprint(initiate_chat_views(socketio, app.extensions['chat_queue']))
    app.register_blueprint(workspace_views)
    app.register_blueprint(agent_views)
    app.register_blueprint(utils_views)

    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"\t{host}{':' if port else ''}{port}{rule}")

    return app, socketio
