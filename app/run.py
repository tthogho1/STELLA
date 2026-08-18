import os

# Anchor relative paths (notably SQLITE_DB_PATH) to this file's own directory regardless
# of the caller's cwd, matching `stella serve`, which chdirs here before running this
# script. Has to happen before any import that constructs the database.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import gevent.monkey
gevent.monkey.patch_all()

from dotenv import load_dotenv

# Loaded before importing app.server: it (and stella_core underneath it) reads
# configuration from the environment at import time.
load_dotenv()

from app.server import create_app
import argparse

# Argument parser (Used for CLI)
parser = argparse.ArgumentParser(description='Run STELLA server.')
parser.add_argument('--host', type=str, default=os.getenv('HOST'), help='Host address for the STELLA server')
parser.add_argument('--port', type=int, default=os.getenv('PORT'), help='Host port for the STELLA server')
args = parser.parse_args()

# Create app
app, socketio = create_app(args.host, args.port)

if __name__ == '__main__':
    socketio.run(app, host=args.host, port=args.port, use_reloader=False)
