"""
Scanning a directory rather than a single file.

This iterates instead of delegating one child per file, because STELLA cannot fan out to
the same agent repeatedly: the action map holds one entry per agent id and duplicate
selections are collapsed. Walking a directory is not a job that needs a model anyway.
"""
import json

import pytest


@pytest.fixture
def agent(monkeypatch, tmp_path):
    source, out = tmp_path / "src", tmp_path / "out"
    (source / "com" / "example").mkdir(parents=True)
    monkeypatch.setenv("SPEC_SOURCE_ROOT", str(source))
    monkeypatch.setenv("SPEC_OUTPUT_ROOT", str(out))

    import importlib
    import stella_agents.CustomAgents.MethodSpecAgent as spec_module
    importlib.reload(spec_module)
    import stella_agents.CustomAgents.RepositorySpecAgent as module
    importlib.reload(module)
    return module.RepositorySpecAgent(), module, source, out


class TestCollecting:
    def test_it_finds_source_files_at_any_depth(self, agent):
        a, _, source, _ = agent
        (source / "A.java").write_text("class A {}")
        (source / "com" / "example" / "B.java").write_text("class B {}")

        assert a._collect(".") == ["A.java", "com/example/B.java"]

    def test_other_file_types_are_ignored(self, agent):
        a, _, source, _ = agent
        (source / "A.java").write_text("class A {}")
        (source / "README.md").write_text("# not source")
        (source / "pom.xml").write_text("<project/>")

        assert a._collect(".") == ["A.java"]

    def test_build_output_is_not_walked(self, agent):
        a, _, source, _ = agent
        (source / "A.java").write_text("class A {}")
        (source / "target").mkdir()
        (source / "target" / "Generated.java").write_text("class Generated {}")

        assert a._collect(".") == ["A.java"]

    def test_the_order_is_stable(self, agent):
        """So a re-scan produces the same index rather than a reshuffled one."""
        a, _, source, _ = agent
        for name in ("C.java", "A.java", "B.java"):
            (source / name).write_text("class X {}")

        assert a._collect(".") == ["A.java", "B.java", "C.java"]

    def test_a_subdirectory_can_be_named(self, agent):
        a, _, source, _ = agent
        (source / "A.java").write_text("class A {}")
        (source / "com" / "example" / "B.java").write_text("class B {}")

        assert a._collect("com/example") == ["com/example/B.java"]

    def test_traversal_is_refused(self, agent):
        a, _, _, _ = agent

        with pytest.raises(ValueError, match="outside the source root"):
            a._collect("../../..")

    def test_a_missing_directory_says_so(self, agent):
        a, _, _, _ = agent

        with pytest.raises(NotADirectoryError):
            a._collect("nope")


class TestFailuresAreRecordedNotRaised:
    """One unreadable file must not lose the specifications of the others."""

    def test_an_oversized_file_is_reported(self, agent, monkeypatch):
        a, module, source, _ = agent
        monkeypatch.setattr(module, "MAX_SOURCE_CHARS", 10)
        (source / "Big.java").write_text("x" * 100)

        result = a._document_one(None, "Big.java")

        assert result["source"] == "Big.java"
        assert "over the 10" in result["error"]

    def test_a_missing_file_is_reported(self, agent):
        a, _, _, _ = agent

        result = a._document_one(None, "Gone.java")

        assert "could not be read" in result["error"]


class TestIndex:
    RESULTS = [
        {"relative": "A.java", "class_name": "A", "methods": [{"method_name": "m"}],
         "written_to": "A.java.spec.json"},
        {"source": "B.java", "error": "the model's answer was not the requested JSON"},
    ]

    def test_it_lists_what_was_covered_and_what_failed(self, agent):
        a, _, _, out = agent

        name = a._write_index("com/example", self.RESULTS)
        document = json.loads((out / name).read_text())

        assert document["scanned"] == "com/example"
        assert document["files"][0] == {"source": "A.java", "spec": "A.java.spec.json",
                                        "class_name": "A", "methods": 1, "error": None}
        assert document["files"][1]["error"].startswith("the model's answer")

    def test_the_digest_names_both(self, agent):
        a, _, _, _ = agent

        digest = a._digest("com/example", self.RESULTS, "index.json", skipped=0)

        assert "documented 1 file(s), 1 method(s)" in digest
        assert "- A.java (A, 1 method(s))" in digest
        assert "! B.java:" in digest
        assert "do not re-scan" in digest.lower()

    def test_the_digest_says_when_files_were_left_out(self, agent):
        a, _, _, _ = agent

        digest = a._digest("com/example", self.RESULTS, "index.json", skipped=7)

        assert "7 further file(s) were not scanned" in digest

    def test_the_digest_does_not_carry_the_findings(self, agent):
        """It goes into memories, which are re-sent on every later agent call."""
        a, _, _, _ = agent
        big = [{"relative": "A.java", "class_name": "A", "written_to": "A.spec.json",
                "methods": [{"method_name": f"m{i}", "findings": [{"detail": "x" * 200}],
                             "uncertainties": []} for i in range(20)]}]

        digest = a._digest("com/example", big, "index.json", skipped=0)

        assert "x" * 200 not in digest
        assert len(digest) < 400
