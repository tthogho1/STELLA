"""
Where the CLI connects and over what.

StellaClient has always taken an `ssl` flag but nothing passed it, so the CLI could only
speak plain http -- sending the password on /login and the JWT on every request after it
in the clear. Anything other than localhost needs this to be settable.
"""
import json

import pytest

from cli.client.stella_client import StellaClient


def _client(host="localhost", port=5001, ssl=False):
    client = StellaClient.__new__(StellaClient)
    client.host, client.port, client.ssl = host, port, ssl
    return client


class TestUrls:
    def test_plain_http_by_default(self):
        assert _client().compose_url("ping") == "http://localhost:5001/ping"

    def test_https_when_enabled(self):
        assert _client("example.com", 5001, True).compose_url("ping") \
            == "https://example.com:5001/ping"

    def test_a_standard_port_can_be_omitted(self):
        """Behind a reverse proxy on 443 there is no port to give. The old form
        interpolated None and produced "https://example.comNone/ping"."""
        url = _client("example.com", None, True).compose_url("ping")

        assert url == "https://example.com/ping"
        assert "None" not in url

    def test_the_socket_shares_the_scheme(self):
        """A wss socket against an https server, not one of each."""
        client = StellaClient.__new__(StellaClient)
        client.host, client.port, client.ssl = "example.com", None, True
        assert client.compose_url("chat/connect") == "https://example.com/chat/connect"


class TestConfig:
    def _load(self, tmp_path, config):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        return json.loads(path.read_text())

    def test_the_shipped_config_parses_and_has_the_key(self):
        import os
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "cli", "config.json")) as f:
            config = json.load(f)

        assert set(config) >= {"host", "port", "ssl"}
        assert config["ssl"] is False, "the default has to stay plain http on localhost"

    def test_a_config_without_ssl_still_works(self, tmp_path):
        """An install predating the key must not break."""
        config = self._load(tmp_path, {"host": "h", "port": 1})

        assert config.get("ssl", False) is False

    def test_a_config_without_a_port_still_works(self, tmp_path):
        config = self._load(tmp_path, {"host": "example.com", "ssl": True})

        assert config.get("port") is None
