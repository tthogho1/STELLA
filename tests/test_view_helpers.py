"""
The shared "load an owned object or fail" helper.

Thirteen routes go through it, so the status codes it picks are the API's contract. The
distinction that matters is between a row that is not there and a lookup that failed:
reporting a locked database as 404 tells the user their workspace does not exist, and
puts the driver's error text in the response body.
"""
import pytest

from stella_core.db.errors import NotFound


@pytest.fixture
def helper():
    from app.utils.view_helpers import get_owned_or_404
    return get_owned_or_404


class _Owned:
    def __init__(self, owner):
        self.owner = owner


def _call(helper, app, *args, **kwargs):
    """jsonify() needs an application context, and the helper builds the response itself,
    so the call has to happen inside one -- not just the reading of the result."""
    with app.app_context():
        obj, err = helper(*args, **kwargs)
        if err is None:
            return obj, None
        body, code = err
        return obj, (code, body.get_json()["msg"])


@pytest.fixture
def app():
    from flask import Flask
    return Flask(__name__)


def test_the_owner_gets_the_object(helper):
    obj, err = helper(lambda _: _Owned("user-1"), "1", "user-1")

    assert err is None
    assert obj.owner == "user-1"


def test_a_missing_row_is_404(helper, app):
    def getter(_):
        raise NotFound("Chat not found")

    obj, err = _call(helper, app, getter, "1", "user-1")

    assert obj is None
    assert err == (404, "Chat not found")


def test_a_broken_database_is_not_reported_as_missing(helper, app):
    """A locked SQLite file or an unreachable mongod is a 500, and its text stays out of
    the response -- as a 404 it told the user the object did not exist."""
    def getter(_):
        raise RuntimeError("database is locked: /var/db/sqlite.db")

    obj, err = _call(helper, app, getter, "1", "user-1")

    code, msg = err
    assert code == 500
    assert "locked" not in msg and "sqlite.db" not in msg


def test_someone_elses_object_is_403(helper, app):
    """Not 401: the caller is authenticated, they just do not own this."""
    obj, err = _call(helper, app, lambda _: _Owned("someone-else"), "1", "user-1",
                     not_owner_msg="User is not the owner of the chat")

    assert obj is None
    assert err == (403, "User is not the owner of the chat")


def test_the_owner_attribute_can_be_named(helper):
    class Task:
        created_by = "user-1"

    obj, err = helper(lambda _: Task(), "1", "user-1", owner_attr="created_by")

    assert err is None


class TestBackendsRaiseNotFound:
    """The helper's 404 branch only works if the backends use the typed error."""

    def test_a_missing_chat(self, db):
        with pytest.raises(NotFound):
            db.get_chat_by_id("999")

    def test_a_missing_workspace(self, db):
        with pytest.raises(NotFound):
            db.get_workspace("999")

    def test_a_missing_task(self, db):
        with pytest.raises(NotFound):
            db.get_task_data("999")

    def test_a_missing_user(self, db):
        with pytest.raises(NotFound):
            db.get_user_by_id("999")


class TestClientTellsTheCodesApart:
    """401 and 403 need different remedies, and the CLI only handled 401."""

    def test_401_asks_for_a_login(self):
        from cli.client.stella_client import StellaClient
        assert "login" in StellaClient.access_error(401).lower()

    def test_403_does_not_ask_for_a_login_alone(self):
        from cli.client.stella_client import StellaClient
        message = StellaClient.access_error(403)
        assert "access" in message.lower()
        assert message != StellaClient.access_error(401)
