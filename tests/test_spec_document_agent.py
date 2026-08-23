"""
Rendering the extracted specifications as a document.

No model is involved -- the specifications are already structured, and putting one here
would only let the document disagree with the JSON it was made from. What matters is that
every claim keeps the line it came from, and that the open questions survive to the end,
because those are the parts nothing in the code settled.
"""
import json

import pytest


SPEC = {
    "source": "com/example/BillingService.java",
    "class_name": "BillingService",
    "methods": [
        {"method_name": "settle", "inputs": ["orderId: String"], "outputs": ["Receipt"],
         "line": 12, "responsibility": "Settles an order",
         "findings": [
             {"kind": "db_operation", "detail": "saves the order", "line": 20,
              "source_line": "orderRepo.save(order);"},
             {"kind": "transaction", "detail": "runs in a transaction", "line": 11,
              "source_line": "@Transactional"}],
         "uncertainties": ["Is an order number unique across the company?"]},
        {"method_name": "cancel", "inputs": [], "outputs": [], "line": 24,
         "responsibility": "Cancels an order", "findings": [], "uncertainties": []},
    ],
}
INDEX = {"scanned": "com/example",
         "files": [{"source": SPEC["source"], "spec": "com/example/BillingService.java.spec.json",
                    "class_name": "BillingService", "methods": 2, "error": None}]}


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEC_OUTPUT_ROOT", str(tmp_path))
    import importlib
    import stella_agents.CustomAgents.MethodSpecAgent as spec_module
    importlib.reload(spec_module)
    import stella_agents.CustomAgents.RepositorySpecAgent as repo_module
    importlib.reload(repo_module)
    import stella_agents.CustomAgents.SpecDocumentAgent as module
    importlib.reload(module)
    return module.SpecDocumentAgent(), module, tmp_path


def _write_output(root, index=INDEX, specs=(SPEC,)):
    (root / "index.json").write_text(json.dumps(index))
    for spec in specs:
        path = root / (spec["source"] + ".spec.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec))


class TestDocument:
    def _render(self, agent):
        a, _, root = agent
        _write_output(root)
        index, specs, problems = a._load()
        return a._render(index, specs, problems)

    def test_every_finding_keeps_its_line_and_its_source(self, agent):
        """A generated specification is only worth having if a claim can be checked."""
        document = self._render(agent)

        assert "| 20 |" in document
        assert "`orderRepo.save(order);`" in document

    def test_findings_are_grouped_in_a_fixed_order(self, agent):
        """So database access is always in the same place, whatever order it came back in."""
        document = self._render(agent)

        assert document.index("Transactions") < document.index("Database access")

    def test_the_signature_is_shown(self, agent):
        document = self._render(agent)

        assert "`settle(orderId: String)` → `Receipt`" in document

    def test_a_method_with_no_return_reads_as_void(self, agent):
        document = self._render(agent)

        assert "`cancel()` → `void`" in document

    def test_open_questions_are_collected_at_the_end(self, agent):
        document = self._render(agent)

        assert "## Open questions" in document
        assert document.count("Is an order number unique across the company?") == 2, \
            "once beside the method and once in the summary"

    def test_the_counts_are_reported(self, agent):
        document = self._render(agent)

        assert "| Methods | 2 |" in document
        assert "| Findings | 2 |" in document
        assert "| Open questions | 1 |" in document

    def test_it_says_the_document_describes_the_code_not_the_intent(self, agent):
        """The distinction the whole exercise depends on."""
        document = self._render(agent)

        assert "what the code does" in document

    def test_a_pipe_in_a_source_line_does_not_break_the_table(self, agent):
        a, _, root = agent
        spec = json.loads(json.dumps(SPEC))
        spec["methods"][0]["findings"][0]["source_line"] = "if (a || b) {"
        _write_output(root, specs=(spec,))
        index, specs, problems = a._load()

        document = a._render(index, specs, problems)

        assert "a \\|\\| b" in document


class TestFailuresAreVisible:
    def test_a_file_that_could_not_be_documented_is_listed(self, agent):
        a, _, root = agent
        index = json.loads(json.dumps(INDEX))
        index["files"].append({"source": "Broken.java", "spec": None, "class_name": None,
                               "methods": 0, "error": "the model's answer was not JSON"})
        _write_output(root, index=index)

        i, specs, problems = a._load()
        document = a._render(i, specs, problems)

        assert "## Not documented" in document
        assert "Broken.java" in document
        assert "| Files not documented | 1 |" in document

    def test_a_missing_spec_file_becomes_a_problem_not_a_crash(self, agent):
        a, _, root = agent
        (root / "index.json").write_text(json.dumps(INDEX))    # index, but no spec file

        _, specs, problems = a._load()

        assert specs == []
        assert problems and "could not be read" in problems[0][1]


class TestWriting:
    def test_nothing_scanned_yet_says_so(self, agent):
        a, _, _ = agent

        assert "scan a source directory first" in a.respond(None, None)

    def test_the_document_is_written_and_the_digest_points_at_it(self, agent):
        a, module, root = agent
        _write_output(root)

        answer = a.respond(None, None)

        assert (root / module.DOCUMENT_NAME).is_file()
        assert module.DOCUMENT_NAME in answer
        assert "1 class(es), 2 method(s), 1 open question(s)" in answer

    def test_the_digest_does_not_repeat_the_document(self, agent):
        """It goes into memories, which are re-sent on every later agent call."""
        a, _, root = agent
        _write_output(root)

        answer = a.respond(None, None)

        assert "orderRepo.save" not in answer
        assert len(answer) < 300
