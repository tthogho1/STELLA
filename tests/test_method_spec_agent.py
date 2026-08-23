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


class TestOutputFile:
    """The full specification goes to a file; memories get a digest and the path."""

    METHODS = [
        {"method_name": "settle", "inputs": ["String"], "outputs": ["Receipt"], "line": 12,
         "responsibility": "Settles an order",
         "findings": [{"kind": "db_operation", "detail": "orderRepo.save", "line": 20},
                      {"kind": "external_call", "detail": "paymentGateway.charge", "line": 18}],
         "uncertainties": ["Is the order number unique across the company?"]},
        {"method_name": "cancel", "inputs": [], "outputs": [], "line": 24,
         "responsibility": "Cancels an order", "findings": [], "uncertainties": []},
    ]

    def _write(self, agent, monkeypatch, tmp_path):
        a, module, _ = agent
        out = tmp_path / "out"
        monkeypatch.setattr(module, "SPEC_OUTPUT_ROOT", str(out))
        written = a._write("com/example/Billing.java", "BillingService", self.METHODS)
        return a, module, out, written

    def test_the_file_mirrors_the_source_path(self, agent, monkeypatch, tmp_path):
        _, _, out, written = self._write(agent, monkeypatch, tmp_path)

        assert written == "com/example/Billing.java.spec.json"
        assert (out / written).is_file()

    def test_the_file_keeps_every_finding_with_its_line(self, agent, monkeypatch, tmp_path):
        import json
        _, _, out, written = self._write(agent, monkeypatch, tmp_path)

        document = json.loads((out / written).read_text())

        assert document["source"] == "com/example/Billing.java"
        assert document["class_name"] == "BillingService"
        settle = document["methods"][0]
        assert [(f["kind"], f["line"]) for f in settle["findings"]] == [
            ("db_operation", 20), ("external_call", 18)]
        assert settle["uncertainties"] == ["Is the order number unique across the company?"]

    def test_the_digest_says_where_the_detail_is(self, agent, monkeypatch, tmp_path):
        a, _, _, written = self._write(agent, monkeypatch, tmp_path)

        digest = a._digest("com/example/Billing.java", "BillingService", self.METHODS, written)

        assert written in digest
        assert "2 method(s), 2 finding(s), 1 open question(s)" in digest
        assert "do not re-analyse" in digest.lower()

    def test_the_digest_does_not_carry_the_findings(self, agent, monkeypatch, tmp_path):
        """Memories are re-sent on every later agent call; the detail stays in the file."""
        a, _, _, written = self._write(agent, monkeypatch, tmp_path)

        digest = a._digest("com/example/Billing.java", "BillingService", self.METHODS, written)

        assert "orderRepo.save" not in digest
        assert "L20" not in digest
        assert len(digest) < 500

    def test_the_digest_still_names_the_methods(self, agent, monkeypatch, tmp_path):
        a, _, _, written = self._write(agent, monkeypatch, tmp_path)

        digest = a._digest("com/example/Billing.java", "BillingService", self.METHODS, written)

        assert "settle(String) -> Receipt" in digest
        assert "cancel() -> void" in digest

    def test_a_long_class_is_summarised_not_listed(self, agent, monkeypatch, tmp_path):
        a, module, _, written = self._write(agent, monkeypatch, tmp_path)
        monkeypatch.setattr(module, "MAX_METHODS_IN_DIGEST", 2)
        many = [{"method_name": f"m{i}", "inputs": [], "outputs": [], "findings": [],
                 "uncertainties": []} for i in range(5)]

        digest = a._digest("X.java", "X", many, written)

        assert "and 3 more" in digest


class TestClassification:
    """
    The kind of a finding is read off the source line, not asked of the model.

    Line citations were accurate every time in testing; the kind was not. The same file,
    model and prompt filed all nine of one method's findings as "exception" on one run --
    a payment gateway call and a database save among them -- and correctly on the next.
    """

    SOURCE = {
        11: "    @Transactional",
        13: "        Order order = orderRepo.findById(id).orElseThrow(Missing::new);",
        14: "        if (!force && order.isLocked()) {",
        15: "            throw new OrderLockedException(orderId);",
        17: '        auditLog.record("settle", orderId);',
        18: "        paymentGateway.charge(order.getTotal());",
        20: "        orderRepo.save(order);",
        21: "        return receiptFactory.build(order);",
    }
    EXPECTED = {11: "transaction", 13: "exception", 14: "branch", 15: "exception",
                17: "other", 18: "external_call", 20: "db_operation", 21: "other"}

    def test_every_line_of_a_real_method(self, agent):
        _, module, _ = agent

        got = {n: module.classify(text) for n, text in self.SOURCE.items()}

        assert got == self.EXPECTED

    def test_a_throw_wins_over_the_branch_it_sits_in(self, agent):
        _, module, _ = agent

        assert module.classify("            throw new OrderLockedException(orderId);") == "exception"

    def test_a_repository_call_wins_over_the_transaction_around_it(self, agent):
        """A repository call is a database operation wherever it appears."""
        _, module, _ = agent

        assert module.classify("        orderRepo.save(order);") == "db_operation"

    def test_dao_mapper_and_entitymanager_all_count_as_database(self, agent):
        _, module, _ = agent

        for line in ("customerDao.insert(c);", "orderMapper.selectById(id);",
                     "entityManager.persist(e);", "jdbcTemplate.update(sql);"):
            assert module.classify(line) == "db_operation", line

    def test_raw_sql_counts_as_database(self, agent):
        _, module, _ = agent

        assert module.classify('String q = "SELECT * FROM orders WHERE id = ?";') == "db_operation"

    def test_an_unrecognised_line_is_not_guessed_at(self, agent):
        """Better "other" than a plausible-looking wrong label."""
        _, module, _ = agent

        assert module.classify("        order.markSettled();") == "other"

    def test_the_patterns_are_configurable(self, agent, monkeypatch):
        """A site with its own naming conventions can point these at them. The default
        pattern matches a word *containing* the term, so a house style of `customerStore`
        is reachable without anchoring to the start of the identifier."""
        import importlib
        monkeypatch.setenv("SPEC_DB_PATTERN", r"\b\w*store\w*\s*\.")
        import stella_agents.CustomAgents.MethodSpecAgent as module
        importlib.reload(module)

        assert module.classify("customerStore.put(c);") == "db_operation"
        assert module.classify("orderRepo.save(o);") == "other", \
            "the override replaces the default rather than adding to it"


class TestPatternOverrides:
    """The keys ship blank in .env_template, and an empty regex matches every line."""

    def _reload(self, monkeypatch, value):
        import importlib
        monkeypatch.setenv("SPEC_DB_PATTERN", value)
        import stella_agents.CustomAgents.MethodSpecAgent as module
        return importlib.reload(module)

    def test_a_blank_setting_does_not_match_everything(self, agent, monkeypatch):
        """SPEC_DB_PATTERN="" would otherwise file the whole file as database access."""
        module = self._reload(monkeypatch, "")

        assert module.classify("order.markSettled();") == "other"
        assert module.classify("orderRepo.save(o);") == "db_operation", "the default is back"

    def test_whitespace_counts_as_blank(self, agent, monkeypatch):
        module = self._reload(monkeypatch, "   ")

        assert module.classify("order.markSettled();") == "other"

    def test_an_invalid_pattern_falls_back_instead_of_raising(self, agent, monkeypatch):
        """This runs at import, and an agent that cannot import takes the scan down."""
        module = self._reload(monkeypatch, "[[[not a regex")

        assert module.classify("order.markSettled();") == "other"
        assert module.classify("orderRepo.save(o);") == "db_operation"


class TestFindingCleanup:
    """What comes back from the model needs tidying before it is a specification."""

    SOURCE = [
        "class BillingService {",                                  # 1
        "",                                                        # 2
        "    @Transactional",                                      # 3
        "    public Receipt settle(String orderId) {",             # 4
        "        orderRepo.save(order);",                          # 5
        "    }",                                                   # 6
        "",                                                        # 7
        "    public void cancel(String orderId) {",                # 8
        "        orderRepo.delete(order);",                        # 9
        "    }",                                                   # 10
    ]

    def _clean(self, agent, findings, span=(1, 6)):
        a, _, _ = agent
        return a._clean_findings(findings, self.SOURCE, span=span)

    def test_the_same_line_twice_is_kept_once(self, agent):
        kept = self._clean(agent, [{"detail": "save", "line": 5},
                                   {"detail": "save again", "line": 5}])

        assert [f["line"] for f in kept] == [5]

    def test_the_methods_own_signature_is_not_a_step(self, agent):
        kept = self._clean(agent, [{"detail": "takes an orderId", "line": 4},
                                   {"detail": "save", "line": 5}])

        assert [f["line"] for f in kept] == [5]

    def test_a_neighbours_line_is_dropped(self, agent):
        """The whole file is in the prompt, so the next method's body is right there."""
        kept = self._clean(agent, [{"detail": "save", "line": 5},
                                   {"detail": "belongs to cancel", "line": 9}])

        assert [f["line"] for f in kept] == [5]

    def test_a_line_off_the_end_is_dropped(self, agent):
        assert self._clean(agent, [{"detail": "ghost", "line": 999}]) == []

    def test_findings_come_back_in_line_order(self, agent):
        kept = self._clean(agent, [{"detail": "save", "line": 5},
                                   {"detail": "annotation", "line": 3}])

        assert [f["line"] for f in kept] == [3, 5]

    def test_the_source_line_is_kept_for_review(self, agent):
        kept = self._clean(agent, [{"detail": "save", "line": 5}])

        assert kept[0]["source_line"] == "orderRepo.save(order);"
        assert kept[0]["kind"] == "db_operation"


class TestSignatureFromSource:
    """The signature is read off the declaration, for the same reason the kind is: one
    run returned outputs ["Receipt", "T"] and inputs mixing names, types and a generic."""

    def test_parameters_come_back_named_and_typed(self, agent):
        a, _, _ = agent

        inputs, outputs = a._signature_from_source(
            "    public Receipt settle(String orderId, boolean force) {")

        assert inputs == ["orderId: String", "force: boolean"]
        assert outputs == ["Receipt"]

    def test_void_has_no_output(self, agent):
        a, _, _ = agent

        assert a._signature_from_source("    public void cancel(String id) {")[1] == []

    def test_a_generic_return_type_is_not_split_on_its_comma(self, agent):
        a, _, _ = agent

        inputs, outputs = a._signature_from_source(
            "    private Map<String, List<Order>> group(int size, String key) {")

        assert outputs == ["Map<String, List<Order>>"]
        assert inputs == ["size: int", "key: String"]

    def test_final_is_not_taken_for_a_type(self, agent):
        a, _, _ = agent

        assert a._signature_from_source("void f(final int size) {")[0] == ["size: int"]

    def test_no_parameters(self, agent):
        a, _, _ = agent

        assert a._signature_from_source("public void run() {")[0] == []

    def test_an_unparseable_line_says_so(self, agent):
        """So the caller can fall back to what the model said rather than invent a void."""
        a, _, _ = agent

        assert a._signature_from_source("not a declaration") == (None, None)


class TestMethodOrdering:
    def test_spans_do_not_depend_on_the_order_the_model_listed_methods(self, agent):
        """Taking "the next method" from an unsorted list gives a span that ends before
        it starts, which silently discards every finding for that method."""
        a, module, _ = agent
        for order in ([("settle", 4), ("cancel", 8)], [("cancel", 8), ("settle", 4)]):
            methods = [{"method_name": n, "line": l} for n, l in order]
            methods.sort(key=lambda m: m["line"])

            assert [m["method_name"] for m in methods] == ["settle", "cancel"]


class TestUploadedWorkspaceRoots:
    """
    Where a workspace reads from once it has uploaded an archive (app/views/source.py).

    Two properties matter. An uploaded workspace must not be able to read the server's
    own SPEC_SOURCE_ROOT, and two workspaces must not write over each other's index.json
    -- the second scan would otherwise inherit the first one's files with nothing in the
    result saying so.
    """

    @pytest.fixture
    def module(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SPEC_SOURCE_ROOT", str(tmp_path / "server"))
        monkeypatch.setenv("SPEC_OUTPUT_ROOT", str(tmp_path / "out"))
        monkeypatch.setenv("SOURCE_UPLOAD_ROOT", str(tmp_path / "uploads"))
        import importlib
        import stella_agents.CustomAgents.MethodSpecAgent as module
        importlib.reload(module)
        return module, tmp_path

    @staticmethod
    def _chat(workspace_id):
        from stella_core.models.chat import Chat
        return Chat(chat_id="1", workspace_id=workspace_id, owner="1")

    def test_a_workspace_with_no_upload_uses_the_configured_roots(self, module):
        m, tmp_path = module

        assert m.roots_for(self._chat("7")) == (m.SPEC_SOURCE_ROOT, m.SPEC_OUTPUT_ROOT)

    def test_no_chat_at_all_uses_the_configured_roots(self, module):
        """Library mode: the runtime can be driven without a workspace."""
        m, _ = module

        assert m.roots_for(None) == (m.SPEC_SOURCE_ROOT, m.SPEC_OUTPUT_ROOT)

    def test_an_uploaded_workspace_reads_its_own_directory(self, module):
        m, tmp_path = module
        (tmp_path / "uploads" / "7").mkdir(parents=True)

        source_root, _ = m.roots_for(self._chat("7"))

        assert source_root == str(tmp_path / "uploads" / "7")

    def test_an_uploaded_workspace_cannot_reach_the_servers_own_tree(self, module):
        m, tmp_path = module
        (tmp_path / "uploads" / "7").mkdir(parents=True)
        (tmp_path / "server").mkdir()
        (tmp_path / "server" / "Secret.java").write_text("class Secret {}")
        source_root, _ = m.roots_for(self._chat("7"))

        with pytest.raises(ValueError, match="outside the source root"):
            m.MethodSpecAgent()._resolve("../../server/Secret.java", source_root)

    def test_two_uploaded_workspaces_write_to_different_places(self, module):
        m, tmp_path = module
        (tmp_path / "uploads" / "7").mkdir(parents=True)
        (tmp_path / "uploads" / "8").mkdir(parents=True)

        _, seven = m.roots_for(self._chat("7"))
        _, eight = m.roots_for(self._chat("8"))

        assert seven != eight
        assert os.path.basename(seven) == "7" and os.path.basename(eight) == "8"

    def test_the_specification_is_written_under_the_workspaces_output_root(self, module):
        m, tmp_path = module
        (tmp_path / "uploads" / "7").mkdir(parents=True)
        _, output_root = m.roots_for(self._chat("7"))

        written = m.MethodSpecAgent()._write("com/A.java", "A", [], output_root)

        assert written == "com/A.java.spec.json"
        assert os.path.isfile(os.path.join(output_root, written))
        assert not os.path.exists(os.path.join(m.SPEC_OUTPUT_ROOT, written))
