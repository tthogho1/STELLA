import os

from flask import Blueprint, jsonify, request
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required
import dotenv
import requests
import zipfile
import io
from flask import current_app

'''
To use local webhooks, you need to install the Stripe CLI:
stripe listen --forward-to localhost:5001/webhook
'''

dotenv.load_dotenv()

webhook_secret = os.environ.get("WEBHOOK_SECRET")

agent_views = Blueprint('agent_views', __name__)

# Must match DOWNLOADED_AGENTS_DIR in app/server.py: that is the directory AgentStorage
# actually scans for runtime-installed packages, so extracting anywhere else would leave
# a downloaded agent invisible until someone noticed and moved it by hand.
DOWNLOADED_AGENTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'agents'))

# Where /agent/download fetches packages from. The URL used to be hardcoded here while
# app/.env_template advertised PACKAGE_MANAGER_URL as the setting for it, so pointing the
# server at a different index -- a private one, or a local mirror -- silently did nothing.
# The default is the address that was hardcoded, so an existing install is unaffected.
DEFAULT_PACKAGE_MANAGER_URL = "https://download-package-d6iaqsbjgq-uc.a.run.app"


# A download should not hold a worker indefinitely if the index stops responding.
PACKAGE_DOWNLOAD_TIMEOUT = 30


def package_manager_url():
    """Read per call, so changing it does not need a restart."""
    return (os.getenv("PACKAGE_MANAGER_URL") or DEFAULT_PACKAGE_MANAGER_URL).rstrip("/")


# A package is a handful of .py files. Nothing legitimate needs more than this, and
# without a ceiling a 199 KB download expands to 200 MB -- measured, deflate reaches
# about 1000:1 on repetitive content.
MAX_PACKAGE_UNCOMPRESSED_BYTES = 50 * 1024 * 1024


def _safe_members(archive):
    """
    The entries of `archive` that are safe to extract, or ValueError explaining why not.

    ZipFile.extract() already strips leading slashes and ".." components, so an entry
    cannot escape the target directory. It does so silently though, which quietly moves
    a file somewhere the package did not intend -- better to refuse the archive and say
    so. The size ceiling is the part that actually protects anything.
    """
    members = []
    total = 0
    for info in archive.infolist():
        name = info.filename
        if "__MACOSX/" in name:
            continue
        if name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/"):
            raise ValueError(f"the archive contains an out-of-tree path ({name})")
        total += info.file_size
        if total > MAX_PACKAGE_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"the archive expands to more than "
                f"{MAX_PACKAGE_UNCOMPRESSED_BYTES // (1024 * 1024)} MB")
        members.append(name)
    return members


@agent_views.route('/agent/download', methods=['GET'])
@jwt_required()
def download_package():
    package_name = request.args.get('query')
    version = request.args.get('version')

    if not package_name:
        return jsonify({"error": "Missing param 'query' "}), 400

    url = package_manager_url()
    # Parameters for the API request
    params = {
        "query": package_name,  # Example query
        "version": version      # Set to None or omit if you want the latest version
    }

    try:
        # A configurable URL means an unreachable or wrong one is a normal thing to hit,
        # not a bug: without this the connection error escapes as a 500 with a stack
        # trace in the body.
        response = requests.get(url, params=params, timeout=PACKAGE_DOWNLOAD_TIMEOUT)
    except requests.RequestException as e:
        print(f"[AGENT] !! Package manager at {url} could not be reached: {e}")
        return jsonify({"msg": f"Could not reach the package manager at {url}"}), 502

    if response.status_code == 200:
        # Save the downloaded file
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            filename = content_disposition.split('filename=')[1].strip('"')
        else:
            filename = "package.zip"  # Fallback filename if header is not set
        
        downloaded_version = filename.removesuffix(".zip")

        try:
            z = zipfile.ZipFile(io.BytesIO(response.content))
            members = _safe_members(z)
        except zipfile.BadZipFile:
            return jsonify({"msg": f"{package_name} is not a valid zip archive"}), 502
        except ValueError as e:
            return jsonify({"msg": f"Refused to install {package_name}: {e}"}), 502

        for file in members:
            z.extract(file, path=DOWNLOADED_AGENTS_DIR)

        return f"Successfully installed {package_name}:{downloaded_version}", response.status_code
    elif response.status_code == 404:
        return f"Package not found: {package_name}:{version}", response.status_code
    else:
        return f"Failed to download package: {package_name}:{version}, {response.text}", response.status_code
    

@agent_views.route('/agent', methods=['GET'])
@jwt_required()
def list_agents():
    """
    Every agent this server has loaded, so /add has something to choose from.

    Without this the agent_id has to be known in advance -- /status only shows what a
    workspace already has, not what is available to add.
    """
    agent_storage = current_app.extensions['agent_storage']

    agents = sorted(
        ({
            "agent_id": agent.agent_id,
            "name": agent.name,
            "short_description": agent.short_description,
            # A leaf does the work itself; one with connections delegates further down.
            "delegates_to": sorted(agent.connections_available or {}),
        } for agent in agent_storage),
        key=lambda a: a["agent_id"],
    )

    return jsonify({"agents": agents, "count": len(agents)}), 200


@agent_views.route('/agent/reload', methods=['get'])
@jwt_required()
def reload_agents():
    """
    Reloads the agents available in agent storage.
    :return:
    """
    agent_storage = current_app.extensions['agent_storage']

    # Reload the agents
    try:
        agent_storage.reload()
    except Exception as e:
        return jsonify({"msg": f"Critical error occurred when reloading agents. Please restart the server. ({str(e)})"}), 500

    return jsonify({"msg": "Agents reloaded"}), 200
