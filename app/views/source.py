"""
Getting source code onto the server so the spec agents can read it.

The spec agents read from a directory on the server's own disk, which is exactly right
when the server runs on the machine holding the code and useless when it does not. This
is the way code reaches a remote server: one zip per workspace, unpacked into a directory
of its own, replacing whatever that workspace had before.

Nothing here is ever executed, and that is a property worth keeping rather than assuming.
An archive unpacked into a directory AgentStorage scans would be imported on the next
GET /agent/reload, which turns "upload a zip" into "run code on this server" -- so the
upload root is checked against those directories, and uploads are refused outright if it
ever overlaps one. Both paths are configurable, which is what makes the check necessary.
"""
import io
import os
import re
import shutil
import zipfile

from flask import Blueprint, jsonify, request, send_file
from flask_jwt_extended import get_jwt_identity, jwt_required

from app.db import db
from app.utils.archives import extract_members, safe_members
from app.utils.view_helpers import get_owned_or_404
from app.views.agent import DOWNLOADED_AGENTS_DIR

source_views = Blueprint('source_views', __name__)

# Where an uploaded archive is unpacked, one directory per workspace. Must match
# SOURCE_UPLOAD_ROOT in stella_agents/CustomAgents/MethodSpecAgent.py -- that is the
# other half of this feature, and the two find each other by this path alone.
#
# Relative to the CWD, like SQLITE_DB_PATH and the agents directory, so it lands under
# app/ when the server is started the documented way (`stella serve` chdirs there).
SOURCE_UPLOAD_ROOT = os.path.abspath(os.getenv("SOURCE_UPLOAD_ROOT", "uploads"))

# What one workspace may store. A source tree is text: 50 MB is a large repository once
# the build output and the .git directory are left out, and the ceiling is what stops a
# zip bomb -- deflate reaches about 1000:1 on repetitive content.
MAX_UNCOMPRESSED_BYTES = int(os.getenv("SOURCE_UPLOAD_MAX_BYTES", str(50 * 1024 * 1024)))

# A separate ceiling because size alone does not bound the work: every file still has to
# be written, walked and considered, and a million empty files weigh nothing.
MAX_FILES = int(os.getenv("SOURCE_UPLOAD_MAX_FILES", "2000"))

# What the request body may be before Flask rejects it, applied in create_app. The
# archive is compressed, so this is deliberately smaller than the uncompressed ceiling.
MAX_REQUEST_BYTES = int(os.getenv("SOURCE_UPLOAD_MAX_REQUEST_BYTES", str(20 * 1024 * 1024)))

# Where the spec agents write. Must match SPEC_OUTPUT_ROOT and the directory layout in
# roots_for() (stella_agents/CustomAgents/MethodSpecAgent.py): a workspace that uploaded
# an archive writes under workspaces/<id>/, and one that did not writes to the root
# itself. tests/test_source_upload.py::TestTheTwoHalvesAgree pins the two together --
# they are in different packages and can only agree by convention.
SPEC_OUTPUT_ROOT = os.path.abspath(os.getenv("SPEC_OUTPUT_ROOT", "spec_output"))
WORKSPACE_OUTPUT_DIR = "workspaces"

# A download builds the zip in memory before sending it. Specifications are text and an
# index; this is a ceiling on a runaway output directory rather than a real expectation.
MAX_DOWNLOAD_BYTES = int(os.getenv("SPEC_DOWNLOAD_MAX_BYTES", str(50 * 1024 * 1024)))

# A workspace id becomes a directory name. Ids are integers on SQLite and hex on
# MongoDB; anything else is not an id this server issued.
SAFE_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _agent_scan_dirs():
    """The directories AgentStorage imports from, which nothing uploaded may land in."""
    import stella_agents
    return [os.path.abspath(os.path.dirname(os.path.abspath(stella_agents.__file__))),
            os.path.abspath(DOWNLOADED_AGENTS_DIR)]


def _misconfiguration():
    """Why uploads are refused, or None. See the module docstring."""
    root = SOURCE_UPLOAD_ROOT
    for scanned in _agent_scan_dirs():
        if root == scanned or root.startswith(scanned + os.sep) \
                or scanned.startswith(root + os.sep):
            return (f"SOURCE_UPLOAD_ROOT ({root}) overlaps a directory agents are "
                    f"imported from ({scanned}). Uploaded files would be executed on the "
                    f"next agent reload. Point it somewhere else and restart the server.")
    return None


def workspace_dir(workspace_id):
    """The directory holding this workspace's source, whether or not it exists yet."""
    if not SAFE_WORKSPACE_ID.match(str(workspace_id)):
        raise ValueError(f"{workspace_id} is not a usable workspace id")
    target = os.path.abspath(os.path.join(SOURCE_UPLOAD_ROOT, str(workspace_id)))
    if not target.startswith(SOURCE_UPLOAD_ROOT + os.sep):
        raise ValueError(f"{workspace_id} is not a usable workspace id")
    return target


def spec_dir(workspace_id):
    """
    Where this workspace's specifications are, whether or not they exist yet.

    The other half of roots_for(): a workspace reading an uploaded archive is isolated,
    output included, and one reading the server's own SPEC_SOURCE_ROOT writes to the
    shared root. That shared root is single-tenant by nature -- every workspace without
    an upload reads and writes the same files, and this endpoint hands them out. Upload
    an archive to get a directory of your own.
    """
    if os.path.isdir(workspace_dir(workspace_id)):
        return os.path.join(SPEC_OUTPUT_ROOT, WORKSPACE_OUTPUT_DIR, str(workspace_id))
    return SPEC_OUTPUT_ROOT


def _summarise(directory):
    """What is on disk for a workspace: file count, bytes, and the top-level entries."""
    files = 0
    total = 0
    for current, _, names in os.walk(directory):
        for name in names:
            files += 1
            try:
                total += os.path.getsize(os.path.join(current, name))
            except OSError:
                pass
    return {"files": files, "bytes": total,
            "entries": sorted(os.listdir(directory)) if os.path.isdir(directory) else []}


def _owned_workspace():
    """(workspace_dir, None) for a caller who owns the workspace, or (None, response)."""
    refusal = _misconfiguration()
    if refusal:
        print(f"[SOURCE] !! {refusal}")
        return None, (jsonify({"msg": refusal}), 503)

    workspace_id = request.view_args["workspace_id"]
    _, err = get_owned_or_404(db.get_workspace, workspace_id, get_jwt_identity(),
                              not_owner_msg="User is not the owner of the workspace")
    if err:
        return None, err

    try:
        return workspace_dir(workspace_id), None
    except ValueError as e:
        return None, (jsonify({"msg": str(e)}), 400)


@source_views.route('/workspace/<workspace_id>/source', methods=['POST'])
@jwt_required()
def upload_source(workspace_id):
    """
    Replaces this workspace's source with the contents of an uploaded zip.

    Replaces rather than merges: a scan walks whatever is on disk, so files left behind
    by an earlier upload would be documented as part of the new one, and nothing in the
    result would say they were stale.
    """
    destination, err = _owned_workspace()
    if err:
        return err

    upload = request.files.get('file')
    if upload is None:
        return jsonify({"msg": "Missing 'file' in the request"}), 400

    try:
        archive = zipfile.ZipFile(upload.stream)
        members = safe_members(archive, MAX_UNCOMPRESSED_BYTES, MAX_FILES)
    except zipfile.BadZipFile:
        return jsonify({"msg": f"{upload.filename or 'the upload'} is not a zip archive"}), 400
    except ValueError as e:
        return jsonify({"msg": f"Refused the archive: {e}"}), 400

    if not members:
        return jsonify({"msg": "The archive holds no files"}), 400

    # Everything below writes to `destination`, which workspace_dir() has already
    # confined to SOURCE_UPLOAD_ROOT. The tree is removed first -- see the docstring.
    try:
        if os.path.isdir(destination):
            shutil.rmtree(destination)
        os.makedirs(destination, exist_ok=True)
        written = extract_members(archive, members, destination)
    except (OSError, ValueError) as e:
        print(f"[SOURCE] !! Could not unpack into {destination}: {e}")
        return jsonify({"msg": f"The archive could not be unpacked ({type(e).__name__})"}), 500

    summary = _summarise(destination)
    print(f"[SOURCE] workspace {workspace_id}: unpacked {written} file(s)")
    return jsonify({
        "msg": f"Unpacked {written} file(s).",
        "files": written,
        # What the user has to name in chat starts with one of these. The archive's own
        # layout is kept as-is (see extract_members), so a zip of a project directory
        # puts a wrapper directory here and this is where they find out.
        "entries": summary["entries"],
    }), 200


@source_views.route('/workspace/<workspace_id>/source', methods=['GET'])
@jwt_required()
def describe_source(workspace_id):
    """What this workspace currently has to analyse."""
    destination, err = _owned_workspace()
    if err:
        return err

    if not os.path.isdir(destination):
        return jsonify({"uploaded": False, "files": 0, "bytes": 0, "entries": []}), 200

    summary = _summarise(destination)
    summary["uploaded"] = True
    return jsonify(summary), 200


@source_views.route('/workspace/<workspace_id>/source', methods=['DELETE'])
@jwt_required()
def delete_source(workspace_id):
    """
    Removes this workspace's uploaded source.

    Worth having as its own call rather than leaving it to an upload of an empty zip:
    while the directory exists the agents read from it instead of SPEC_SOURCE_ROOT, so
    this is how a workspace goes back to the server's own configured tree.
    """
    destination, err = _owned_workspace()
    if err:
        return err

    if not os.path.isdir(destination):
        return jsonify({"msg": "Nothing was uploaded for this workspace."}), 200

    try:
        shutil.rmtree(destination)
    except OSError as e:
        print(f"[SOURCE] !! Could not remove {destination}: {e}")
        return jsonify({"msg": f"Could not remove the source ({type(e).__name__})"}), 500

    return jsonify({"msg": "Removed the uploaded source."}), 200


@source_views.route('/workspace/<workspace_id>/spec', methods=['GET'])
@jwt_required()
def download_spec(workspace_id):
    """
    Sends this workspace's generated specifications back as a zip.

    The counterpart to the upload: on a remote server the agents write to a disk the
    user has no other way of reading, and the chat only ever carries a digest -- by
    design, since memories are re-sent on every later agent call.
    """
    refusal = _misconfiguration()
    if refusal:
        return jsonify({"msg": refusal}), 503

    _, err = get_owned_or_404(db.get_workspace, workspace_id, get_jwt_identity(),
                              not_owner_msg="User is not the owner of the workspace")
    if err:
        return err

    try:
        directory = spec_dir(workspace_id)
    except ValueError as e:
        return jsonify({"msg": str(e)}), 400

    if not os.path.isdir(directory):
        return jsonify({"msg": "Nothing has been generated for this workspace yet."}), 404

    buf = io.BytesIO()
    total = 0
    written = 0
    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            for current, subdirs, names in os.walk(directory):
                subdirs.sort()
                for name in sorted(names):
                    path = os.path.join(current, name)
                    total += os.path.getsize(path)
                    if total > MAX_DOWNLOAD_BYTES:
                        return jsonify({"msg": f"The generated output is larger than "
                                               f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB. "
                                               f"Read it on the server instead."}), 413
                    archive.write(path, os.path.relpath(path, directory))
                    written += 1
    except OSError as e:
        print(f"[SOURCE] !! Could not read {directory}: {e}")
        return jsonify({"msg": f"The output could not be read ({type(e).__name__})"}), 500

    if not written:
        return jsonify({"msg": "Nothing has been generated for this workspace yet."}), 404

    buf.seek(0)
    print(f"[SOURCE] workspace {workspace_id}: sending {written} generated file(s)")
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"stella-spec-{workspace_id}.zip")


@source_views.app_errorhandler(413)
def too_large(_):
    """
    Flask rejects an oversized body before any view runs, and the default page is HTML.

    Registered app-wide from this blueprint because this is the only endpoint that takes
    a body big enough to hit MAX_CONTENT_LENGTH.
    """
    return jsonify({"msg": f"The upload is larger than "
                           f"{MAX_REQUEST_BYTES // (1024 * 1024)} MB."}), 413
