"""
Where /agent/download fetches packages from.

app/.env_template advertised PACKAGE_MANAGER_URL as the setting for this while the view
used a hardcoded address, so pointing a server at a private index or a local mirror
silently did nothing.
"""
import pytest

from app.views.agent import DEFAULT_PACKAGE_MANAGER_URL, package_manager_url


def test_the_environment_wins(monkeypatch):
    monkeypatch.setenv("PACKAGE_MANAGER_URL", "https://packages.internal.example")

    assert package_manager_url() == "https://packages.internal.example"


def test_it_falls_back_to_the_address_that_was_hardcoded(monkeypatch):
    """An install that never set the variable has to keep working unchanged."""
    monkeypatch.delenv("PACKAGE_MANAGER_URL", raising=False)

    assert package_manager_url() == DEFAULT_PACKAGE_MANAGER_URL


def test_an_empty_value_is_treated_as_unset(monkeypatch):
    """app/.env_template ships the key, so a blank one must not produce a request to ''."""
    monkeypatch.setenv("PACKAGE_MANAGER_URL", "")

    assert package_manager_url() == DEFAULT_PACKAGE_MANAGER_URL


def test_a_trailing_slash_does_not_double_up(monkeypatch):
    monkeypatch.setenv("PACKAGE_MANAGER_URL", "https://packages.internal.example/")

    assert package_manager_url() == "https://packages.internal.example"


def test_it_is_read_per_call(monkeypatch):
    """Not captured at import, so a change takes effect without a restart."""
    monkeypatch.setenv("PACKAGE_MANAGER_URL", "https://first.example")
    first = package_manager_url()
    monkeypatch.setenv("PACKAGE_MANAGER_URL", "https://second.example")

    assert first == "https://first.example"
    assert package_manager_url() == "https://second.example"


class TestUnreachableIndex:
    """Now that the URL is configurable, pointing it somewhere wrong is a normal mistake
    rather than a bug -- it must not surface as a 500 with a stack trace."""

    def _download(self, monkeypatch, raises):
        import requests as requests_module
        from flask import Flask
        import app.views.agent as agent_views

        def boom(*args, **kwargs):
            raise raises

        monkeypatch.setattr(agent_views.requests, "get", boom)
        flask_app = Flask(__name__)
        with flask_app.test_request_context('/agent/download?query=some-package'):
            body, status = agent_views.download_package.__wrapped__()
            return body.get_json(), status

    def test_a_refused_connection_is_502(self, monkeypatch):
        import requests
        body, status = self._download(monkeypatch, requests.ConnectionError("refused"))

        assert status == 502
        assert "package manager" in body["msg"]

    def test_a_timeout_is_502(self, monkeypatch):
        import requests
        body, status = self._download(monkeypatch, requests.Timeout("too slow"))

        assert status == 502

    def test_the_driver_error_is_not_echoed_back(self, monkeypatch):
        import requests
        body, _ = self._download(
            monkeypatch, requests.ConnectionError("HTTPConnectionPool(host='10.0.0.1'...)"))

        assert "HTTPConnectionPool" not in body["msg"]
