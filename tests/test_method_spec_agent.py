"""
The source-file specification agent.

Two things here are not about the model. The path arrives from a chat message by way of
an LLM, so it has to be confined to a root; and the outline pass returns methods the
class *calls* as well as the ones it declares, which is checkable against the source.
"""
import os

import pytest


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEC_SOURCE_ROOT", str(tmp_path))
    import importlib
    import stella_agents.CustomAgents.MethodSpecAgent as module
    importlib.reload(module)
    return module.MethodSpecAgent(), module, tmp_path


class TestPathConfinement:
    def test_a_file_under_the_root_resolves(self, agent):
        a, _, root = agent
        (root / "Thing.java").write_text("class Thing {}")

        assert a._resolve("Thing.java") == str(root / "Thing.java")

    def test_a_leading_slash_is_treated_as_relative(self, agent):
        """Models answer with "/src/com/..." even when asked for the path as written."""
        a, _, root = agent
        (root / "Thing.java").write_text("class Thing {}")

        assert a._resolve("/Thing.java") == str(root / "Thing.java")

    def test_quotes_are_stripped(self, agent):
        a, _, root = agent
        (root / "Thing.java").write_text("class Thing {}")

        assert a._resolve('"Thing.java"') == str(root / "Thing.java")

    def test_traversal_is_refused(self, agent):
        a, _, _ = agent

        with pytest.raises(ValueError, match="outside the source root"):
            a._resolve("../../../etc/passwd")

    def test_an_absolute_path_elsewhere_cannot_escape(self, agent):
        a, _, _ = agent

        with pytest.raises((ValueError, FileNotFoundError)):
            a._resolve("/etc/passwd")

    def test_a_directory_is_not_a_file(self, agent):
        a, _, root = agent
        (root / "sub").mkdir()

        with pytest.raises(FileNotFoundError):
            a._resolve("sub")


class TestDeclarationFilter:
    """The outline pass lists called methods too. Their cited line gives them away."""

    SOURCE = """class BillingService {
    public Receipt settle(String orderId) {
        orderRepo.save(order);
        return receiptFactory.build(order);
    }
}"""

    def test_a_declared_method_is_kept(self, agent):
        a, _, _ = agent
        kept = a._declared_only([{"method_name": "settle", "line": 2}], self.SOURCE)

        assert [m["method_name"] for m in kept] == ["settle"]

    def test_a_called_method_is_dropped(self, agent):
        """orderRepo.save is a call on line 3, not a declaration."""
        a, _, _ = agent
        kept = a._declared_only([{"method_name": "save", "line": 3},
                                 {"method_name": "build", "line": 4}], self.SOURCE)

        assert kept == []

    def test_a_line_outside_the_file_is_dropped(self, agent):
        a, _, _ = agent

        assert a._declared_only([{"method_name": "ghost", "line": 999}], self.SOURCE) == []

    def test_a_missing_line_is_dropped(self, agent):
        a, _, _ = agent

        assert a._declared_only([{"method_name": "ghost"}], self.SOURCE) == []


class TestNumbering:
    def test_lines_are_numbered_from_one(self, agent):
        a, _, _ = agent

        numbered = a._number("first\nsecond")

        assert numbered.splitlines()[0].strip().startswith("1: first")
        assert numbered.splitlines()[1].strip().startswith("2: second")


class TestRendering:
    def test_a_finding_keeps_its_line(self, agent):
        a, _, _ = agent
        out = a._render("BillingService", "Billing.java", [
            {"method_name": "settle", "inputs": ["String"], "outputs": ["Receipt"],
             "responsibility": "Settles an order",
             "findings": [{"kind": "db_operation", "detail": "orderRepo.save", "line": 20}],
             "uncertainties": ["Is the order number unique across the company?"]}])

        assert "settle(String) -> Receipt" in out
        assert "[db_operation] L20: orderRepo.save" in out
        assert "[?] Is the order number unique" in out

    def test_a_method_with_no_outputs_reads_as_void(self, agent):
        a, _, _ = agent
        out = a._render("C", "C.java", [{"method_name": "cancel", "inputs": [],
                                         "outputs": [], "findings": [], "uncertainties": []}])

        assert "cancel() -> void" in out
