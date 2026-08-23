"""
Uploading a source tree for the spec agents to read.

This is the one endpoint that takes a file from the user and writes it to the server's
disk, so most of what matters here is what it refuses. The archive checks it shares with
/agent/download are covered in test_package_extraction.py; these are the parts that only
apply to an upload: the prefix a repository zip carries, replacing rather than merging,
and the guard that keeps uploaded files away from the directories agents are imported
from.
"""
import io
import os
import zipfile
from types import SimpleNamespace

import pytest
from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token

OWNER = "1"
WORKSPACE = "7"


def _zip(entries, compress=zipfile.ZIP_STORED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as z:
        for name, content in entries:
            z.writestr(name, content)
    buf.seek(0)
    return buf


@pytest.fixture
def uploads(monkeypatch, tmp_path):
    """The endpoint, wired to a throwaway upload root and a stubbed workspace lookup."""
    root = tmp_path / "uploads"
    monkeypatch.setenv("SOURCE_UPLOAD_ROOT", str(root))
    # Both are read at import. Without this the generated-output tests write into
    # whatever SPEC_OUTPUT_ROOT the developer's own app/.env points at.
    monkeypatch.setenv("SPEC_OUTPUT_ROOT", str(tmp_path / "out"))

    import importlib
    import app.views.source as module
    importlib.reload(module)

    def owned(_getter, workspace_id, user_id, **_kwargs):
        from flask import jsonify
        if user_id != OWNER:
            return None, (jsonify({"msg": "not the owner"}), 403)
        return object(), None

    monkeypatch.setattr(module, "get_owned_or_404", owned)
    # The real `db` is a proxy that raises until a backend is initialised, and it is
    # dereferenced to build the argument before the stub above is ever called.
    monkeypatch.setattr(module, "db", SimpleNamespace(get_workspace=lambda _id: None))

    flask_app = Flask(__name__)
    flask_app.config["JWT_SECRET_KEY"] = "test-key"
    flask_app.config["MAX_CONTENT_LENGTH"] = module.MAX_REQUEST_BYTES
    JWTManager(flask_app)
    flask_app.register_blueprint(module.source_views)

    with flask_app.app_context():
        tokens = {OWNER: create_access_token(identity=OWNER),
                  "2": create_access_token(identity="2")}

    client = flask_app.test_client()

    def post(buf, as_user=OWNER, workspace=WORKSPACE):
        return client.post(f"/workspace/{workspace}/source",
                           headers={"Authorization": f"Bearer {tokens[as_user]}"},
                           data={"file": (buf, "source.zip")},
                           content_type="multipart/form-data")

    def call(method, as_user=OWNER, workspace=WORKSPACE, path="source"):
        return getattr(client, method)(
            f"/workspace/{workspace}/{path}",
            headers={"Authorization": f"Bearer {tokens[as_user]}"})

    return post, call, root, module


class TestUnpacking:
    def test_an_archive_lands_under_the_workspaces_own_directory(self, uploads):
        post, _, root, _ = uploads

        response = post(_zip([("com/example/A.java", "class A {}"), ("pom.xml", "<p/>")]))

        assert response.status_code == 200
        assert response.get_json()["files"] == 2
        assert (root / WORKSPACE / "com" / "example" / "A.java").read_text() == "class A {}"

    def test_another_workspace_is_untouched(self, uploads):
        """Two workspaces analysing different code must not see each other's files."""
        post, _, root, _ = uploads
        post(_zip([("A.java", "class A {}")]), workspace="7")
        post(_zip([("B.java", "class B {}")]), workspace="8")

        assert (root / "7" / "A.java").is_file()
        assert not (root / "7" / "B.java").exists()
        assert (root / "8" / "B.java").is_file()

    def test_the_archives_own_layout_is_kept(self, uploads):
        """A wrapper directory (`orders-main/` from GitHub) is left alone. Dropping it
        is tempting, but it is indistinguishable from an archive of a single package,
        where dropping it would break the path the user names in chat."""
        post, _, root, _ = uploads

        response = post(_zip([("orders-main/pom.xml", "<project/>"),
                              ("orders-main/src/A.java", "class A {}")]))

        assert (root / WORKSPACE / "orders-main" / "src" / "A.java").is_file()
        assert response.get_json()["entries"] == ["orders-main"], \
            "the user has to be told what their paths start with"

    def test_a_single_package_directory_survives(self, uploads):
        """The case that made stripping a wrong answer: `com/` is the only top-level
        entry, and "document com/example" has to keep working."""
        post, _, root, _ = uploads

        post(_zip([("com/example/A.java", "class A {}")]))

        assert (root / WORKSPACE / "com" / "example" / "A.java").is_file()

    def test_an_upload_replaces_what_was_there(self, uploads):
        """A scan walks whatever is on disk, so a file left behind by an earlier upload
        would be documented as part of the new one with nothing saying it was stale."""
        post, _, root, _ = uploads
        post(_zip([("Old.java", "class Old {}")]))

        post(_zip([("New.java", "class New {}")]))

        assert (root / WORKSPACE / "New.java").is_file()
        assert not (root / WORKSPACE / "Old.java").exists()


class TestRefusals:
    def test_a_non_owner_cannot_upload(self, uploads):
        post, _, root, _ = uploads

        assert post(_zip([("A.java", "x")]), as_user="2").status_code == 403
        assert not (root / WORKSPACE).exists()

    def test_something_that_is_not_a_zip(self, uploads):
        post, _, root, _ = uploads

        response = post(io.BytesIO(b"this is not a zip file"))

        assert response.status_code == 400
        assert "not a zip archive" in response.get_json()["msg"]

    def test_a_traversal_writes_nothing(self, uploads):
        post, _, root, _ = uploads

        response = post(_zip([("../../escaped.java", "class Escaped {}")]))

        assert response.status_code == 400
        assert "out-of-tree" in response.get_json()["msg"]
        assert not (root.parent / "escaped.java").exists()

    def test_a_zip_bomb(self, uploads):
        post, _, _, module = uploads
        oversized = b"\0" * (module.MAX_UNCOMPRESSED_BYTES + 1)

        response = post(_zip([("big.java", oversized)], zipfile.ZIP_DEFLATED))

        assert response.status_code == 400
        assert "expands to more than" in response.get_json()["msg"]

    def test_too_many_files(self, uploads, monkeypatch):
        post, _, _, module = uploads
        monkeypatch.setattr(module, "MAX_FILES", 3)

        response = post(_zip([(f"F{i}.java", "x") for i in range(4)]))

        assert response.status_code == 400
        assert "more than 3 files" in response.get_json()["msg"]

    def test_an_empty_archive(self, uploads):
        post, _, _, _ = uploads

        assert post(_zip([])).status_code == 400

    def test_a_request_with_no_file(self, uploads):
        _, _, _, module = uploads
        # Posting nothing at all, rather than through the fixture's multipart helper.
        flask_app = Flask(__name__)
        flask_app.config["JWT_SECRET_KEY"] = "test-key"
        JWTManager(flask_app)
        flask_app.register_blueprint(module.source_views)
        with flask_app.app_context():
            token = create_access_token(identity=OWNER)
        response = flask_app.test_client().post(
            f"/workspace/{WORKSPACE}/source",
            headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 400
        assert "Missing 'file'" in response.get_json()["msg"]

    def test_an_oversized_body_is_json_not_html(self, uploads, monkeypatch):
        """Flask rejects it before any view runs, and the default page is an HTML one."""
        post, _, _, module = uploads

        response = post(io.BytesIO(b"\0" * (module.MAX_REQUEST_BYTES + 1024)))

        assert response.status_code == 413
        assert "larger than" in response.get_json()["msg"]


class TestExecutionGuard:
    """
    Nothing uploaded is ever imported, and that has to stay true by construction rather
    than by nobody having pointed the two settings at the same place.
    """

    def test_an_upload_root_inside_the_agents_directory_is_refused(
            self, monkeypatch, tmp_path):
        import importlib
        import app.views.source as module
        monkeypatch.setenv("SOURCE_UPLOAD_ROOT", str(tmp_path / "uploads"))
        importlib.reload(module)
        monkeypatch.setattr(module, "SOURCE_UPLOAD_ROOT",
                            os.path.join(module.DOWNLOADED_AGENTS_DIR, "uploads"))
        monkeypatch.setattr(module, "get_owned_or_404",
                            lambda *a, **k: (object(), None))
        monkeypatch.setattr(module, "db", SimpleNamespace(get_workspace=lambda _id: None))

        flask_app = Flask(__name__)
        flask_app.config["JWT_SECRET_KEY"] = "test-key"
        JWTManager(flask_app)
        flask_app.register_blueprint(module.source_views)
        with flask_app.app_context():
            token = create_access_token(identity=OWNER)

        response = flask_app.test_client().post(
            f"/workspace/{WORKSPACE}/source",
            headers={"Authorization": f"Bearer {token}"},
            data={"file": (_zip([("A.java", "x")]), "source.zip")},
            content_type="multipart/form-data")

        assert response.status_code == 503
        assert "would be executed" in response.get_json()["msg"]

    def test_a_workspace_id_that_is_not_one_cannot_name_a_directory(self, uploads):
        _, _, _, module = uploads

        with pytest.raises(ValueError):
            module.workspace_dir("../../etc")


class TestDescribeAndDelete:
    def test_nothing_uploaded(self, uploads):
        _, call, _, _ = uploads

        assert call("get").get_json() == {"uploaded": False, "files": 0,
                                          "bytes": 0, "entries": []}

    def test_what_is_there(self, uploads):
        post, call, _, _ = uploads
        post(_zip([("src/A.java", "class A {}"), ("pom.xml", "<project/>")]))

        body = call("get").get_json()

        assert body["uploaded"] is True and body["files"] == 2
        assert body["entries"] == ["pom.xml", "src"]

    def test_deleting_puts_the_workspace_back_on_the_servers_own_tree(self, uploads):
        post, call, root, _ = uploads
        post(_zip([("A.java", "class A {}")]))

        assert call("delete").status_code == 200
        assert not (root / WORKSPACE).exists()

    def test_deleting_nothing_is_not_an_error(self, uploads):
        _, call, _, _ = uploads

        assert call("delete").status_code == 200

    def test_a_non_owner_cannot_delete(self, uploads):
        post, call, root, _ = uploads
        post(_zip([("A.java", "class A {}")]))

        assert call("delete", as_user="2").status_code == 403
        assert (root / WORKSPACE / "A.java").is_file()


class TestDownload:
    """
    Getting the generated specifications back off the server.

    The counterpart to the upload: on a remote server the agents write to a disk the
    user has no other way of reading, and the chat only ever carries a digest.
    """

    @staticmethod
    def _generate(module, workspace, files):
        """Stands in for a scan having run, writing where the agents would have."""
        directory = module.spec_dir(workspace)
        for name, content in files.items():
            path = os.path.join(directory, *name.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as handle:
                handle.write(content)
        return directory

    def test_nothing_generated_yet(self, uploads):
        _, call, _, _ = uploads

        response = call("get", path="spec")

        assert response.status_code == 404
        assert "Nothing has been generated" in response.get_json()["msg"]

    def test_the_generated_tree_comes_back_as_a_zip(self, uploads):
        post, call, _, module = uploads
        post(_zip([("A.java", "class A {}")]))
        self._generate(module, WORKSPACE,
                       {"index.json": "{}", "specification.md": "# Specification",
                        "com/A.java.spec.json": '{"class_name": "A"}'})

        response = call("get", path="spec")

        assert response.status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(response.data))
        assert sorted(archive.namelist()) == ["com/A.java.spec.json", "index.json",
                                              "specification.md"]
        assert archive.read("specification.md").decode() == "# Specification"

    def test_it_is_offered_as_a_file_named_for_the_workspace(self, uploads):
        post, call, _, module = uploads
        post(_zip([("A.java", "class A {}")]))
        self._generate(module, WORKSPACE, {"index.json": "{}"})

        response = call("get", path="spec")

        assert response.headers["Content-Disposition"].endswith(
            f'filename=stella-spec-{WORKSPACE}.zip')

    def test_one_workspace_cannot_fetch_anothers(self, uploads):
        post, call, _, module = uploads
        post(_zip([("A.java", "class A {}")]), workspace="7")
        post(_zip([("B.java", "class B {}")]), workspace="8")
        self._generate(module, "7", {"seven.md": "seven"})
        self._generate(module, "8", {"eight.md": "eight"})

        archive = zipfile.ZipFile(io.BytesIO(call("get", path="spec", workspace="8").data))

        assert archive.namelist() == ["eight.md"]

    def test_a_non_owner_is_refused(self, uploads):
        post, call, _, module = uploads
        post(_zip([("A.java", "class A {}")]))
        self._generate(module, WORKSPACE, {"index.json": "{}"})

        assert call("get", path="spec", as_user="2").status_code == 403

    def test_an_oversized_output_is_refused_rather_than_built(self, uploads, monkeypatch):
        post, call, _, module = uploads
        post(_zip([("A.java", "class A {}")]))
        self._generate(module, WORKSPACE, {"big.md": "x" * 4096})
        monkeypatch.setattr(module, "MAX_DOWNLOAD_BYTES", 1024)

        response = call("get", path="spec")

        assert response.status_code == 413
        assert "larger than" in response.get_json()["msg"]

    def test_an_empty_output_directory_is_not_an_empty_zip(self, uploads):
        """An empty archive downloads fine and then tells the user nothing at all."""
        post, call, _, module = uploads
        post(_zip([("A.java", "class A {}")]))
        os.makedirs(module.spec_dir(WORKSPACE), exist_ok=True)

        assert call("get", path="spec").status_code == 404


class TestTheTwoHalvesAgree:
    """
    app/views/source.py and stella_agents/CustomAgents/MethodSpecAgent.py are in
    different packages -- the agents cannot import the server, and the server must not
    import a particular agent -- so they agree on where files go by convention alone.
    Nothing but this catches one side being changed without the other.
    """

    @pytest.fixture
    def both(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SOURCE_UPLOAD_ROOT", str(tmp_path / "uploads"))
        monkeypatch.setenv("SPEC_OUTPUT_ROOT", str(tmp_path / "out"))
        monkeypatch.setenv("SPEC_SOURCE_ROOT", str(tmp_path / "server"))
        import importlib
        import app.views.source as view
        import stella_agents.CustomAgents.MethodSpecAgent as agent
        importlib.reload(view)
        importlib.reload(agent)
        return view, agent, tmp_path

    @staticmethod
    def _chat(workspace_id):
        from stella_core.models.chat import Chat
        return Chat(chat_id="1", workspace_id=workspace_id, owner=OWNER)

    def test_they_agree_on_where_an_uploaded_workspace_reads_from(self, both):
        view, agent, tmp_path = both
        (tmp_path / "uploads" / WORKSPACE).mkdir(parents=True)

        source_root, _ = agent.roots_for(self._chat(WORKSPACE))

        assert source_root == view.workspace_dir(WORKSPACE)

    def test_they_agree_on_where_an_uploaded_workspace_writes_to(self, both):
        view, agent, tmp_path = both
        (tmp_path / "uploads" / WORKSPACE).mkdir(parents=True)

        _, output_root = agent.roots_for(self._chat(WORKSPACE))

        assert output_root == view.spec_dir(WORKSPACE)

    def test_they_agree_when_there_is_no_upload(self, both):
        view, agent, _ = both

        _, output_root = agent.roots_for(self._chat(WORKSPACE))

        assert output_root == view.spec_dir(WORKSPACE) == agent.SPEC_OUTPUT_ROOT
