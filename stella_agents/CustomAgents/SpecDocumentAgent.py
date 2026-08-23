"""
Turns the extracted specifications into a document someone can read and check.

No model is involved: the specifications are already structured, and rendering them is
formatting. An LLM here would only add a chance of the document disagreeing with the JSON
it was made from.

The layout follows from what a generated specification is worth: every finding carries the
line it came from and the source line itself, so a reviewer can check a claim without
opening the file, and the open questions are collected into one section because those are
the parts nobody has confirmed yet.
"""
import json
import os
import re
from datetime import datetime, timezone

from stella_core.models.agent import Agent
from stella_core.models.chat import Chat
from stella_core.openai_client import OpenAIClient
from stella_core.utils.request_builder import RequestBuilder

from stella_agents.CustomAgents.MethodSpecAgent import (
    KIND_MEANINGS, SPEC_OUTPUT_ROOT, roots_for)
from stella_agents.CustomAgents.RepositorySpecAgent import INDEX_NAME

DOCUMENT_NAME = os.getenv("SPEC_DOCUMENT_NAME", "specification.md")

# Findings are grouped under these headings, in this order, so a reader always finds
# database access in the same place. "other" is last: it is the unclassified remainder.
KIND_ORDER = ["transaction", "branch", "db_operation", "external_call", "exception", "other"]

KIND_LABELS = {
    "db_operation": "Database access",
    "external_call": "Calls out of the system",
    "exception": "Exceptions",
    "transaction": "Transactions",
    "branch": "Branches",
    "other": "Other steps",
}


def _escape(text):
    """Keeps a pipe in a source line from breaking the table it sits in."""
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


class SpecDocumentAgent(Agent):
    """Renders the extracted specifications as a Markdown document."""

    def __init__(self):
        super().__init__(
            agent_id='spec_document_agent',
            name='SPEC_DOCUMENT',
            short_description='Turn the extracted specifications into a Markdown document, '
                              'with every finding traced to a line of source',
            skip_action_selection=True,
            forward_all_memory_entries_to_parent=True,
        )

    @staticmethod
    def _load(output_root=None):
        """
        Reads the index and every specification it names.

        :return: (index, [spec, ...], [problem, ...])
        :raises FileNotFoundError: when nothing has been scanned yet
        """
        output_root = output_root or SPEC_OUTPUT_ROOT
        index_path = os.path.join(output_root, INDEX_NAME)
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)

        specs, problems = [], []
        for entry in index.get("files", []):
            if entry.get("error"):
                problems.append((entry["source"], entry["error"]))
                continue
            path = os.path.join(output_root, entry["spec"])
            try:
                with open(path, encoding="utf-8") as handle:
                    specs.append(json.load(handle))
            except (OSError, json.JSONDecodeError) as e:
                problems.append((entry["source"], f"its specification could not be read ({e})"))
        return index, specs, problems

    @staticmethod
    def _documented_elsewhere(question, class_names):
        """
        The classes this question asks about that this document already covers.

        Used to annotate, not to drop. Deleting on this signal was tried and measured
        against real output: "what should happen if customerService.register(request)
        returns an exception" -- a question only a person can answer -- names a class in
        the scan just as "what the customer service class entails" does, and no rule
        separated the two without also removing the first. Losing a real question is a
        worse outcome than carrying one that turns out to be answerable, so the reader is
        told where to look and decides.

        Matched on letters and digits alone, because a model writes `CustomerRequest` as
        "the customer request" about as often as not.
        """
        key = re.sub(r"[^a-z0-9]", "", question.lower())
        return sorted({name for name in class_names
                       if len(name) > 3 and re.sub(r"[^a-z0-9]", "", name.lower()) in key})

    @staticmethod
    def _method_section(method, out):
        signature = (f"{method['method_name']}"
                     f"({', '.join(method.get('inputs') or [])})")
        returns = ', '.join(method.get('outputs') or []) or 'void'
        out.append(f"#### `{_escape(signature)}` → `{_escape(returns)}`")
        out.append("")
        if method.get("responsibility"):
            out.append(_escape(method["responsibility"]))
            out.append("")

        findings = method.get("findings") or []
        if findings:
            by_kind = {}
            for finding in findings:
                by_kind.setdefault(finding.get("kind", "other"), []).append(finding)

            out.append("| Line | What | Source |")
            out.append("| ---: | --- | --- |")
            for kind in KIND_ORDER:
                for finding in sorted(by_kind.get(kind, []), key=lambda f: f["line"]):
                    out.append(f"| {finding['line']} | **{KIND_LABELS.get(kind, kind)}** "
                               f"— {_escape(finding.get('detail'))} "
                               f"| `{_escape(finding.get('source_line'))}` |")
            out.append("")

        for question in method.get("uncertainties") or []:
            out.append(f"> **Open question.** {_escape(question)}")
            out.append("")

    def _render(self, index, specs, problems):
        generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        methods = sum(len(s.get("methods") or []) for s in specs)
        findings = sum(len(m.get("findings") or [])
                       for s in specs for m in (s.get("methods") or []))
        questions = [(s["source"], m["method_name"], q)
                     for s in specs for m in (s.get("methods") or [])
                     for q in (m.get("uncertainties") or [])]

        out = [
            "# Specification",
            "",
            f"Extracted from `{index.get('scanned', '?')}` on {generated}.",
            "",
            "> Generated from the source, and describing **what the code does** rather than "
            "what it is supposed to do. Every entry cites the line it came from so it can "
            "be checked. The open questions at the end are the parts nothing in the code "
            "settled.",
            "",
            "| | |", "| --- | ---: |",
            f"| Files documented | {len(specs)} |",
            f"| Methods | {methods} |",
            f"| Findings | {findings} |",
            f"| Open questions | {len(questions)} |",
        ]
        if problems:
            out.append(f"| Files not documented | {len(problems)} |")
        out.append("")

        out.append("## Classes")
        out.append("")
        for spec in sorted(specs, key=lambda s: s["source"]):
            out.append(f"### `{_escape(spec.get('class_name', '?'))}`")
            out.append("")
            out.append(f"`{_escape(spec['source'])}`")
            out.append("")
            for method in sorted(spec.get("methods") or [],
                                 key=lambda m: m.get("line") or 0):
                self._method_section(method, out)

        if questions:
            out.append("## Open questions")
            out.append("")
            out.append("Nothing in the code answered these. They need a person who knows "
                       "the intended behaviour.")
            out.append("")
            out.append("| File | Method | Question | Covered here |")
            out.append("| --- | --- | --- | --- |")
            class_names = {s.get("class_name") for s in specs if s.get("class_name")}
            for source, method_name, question in questions:
                named = self._documented_elsewhere(question, class_names)
                reference = ", ".join(f"`{_escape(n)}`" for n in named) if named else ""
                out.append(f"| `{_escape(source)}` | `{_escape(method_name)}` "
                           f"| {_escape(question)} | {reference} |")
            out.append("")
            out.append("A class in the last column is documented above: read that section "
                       "before asking anyone.")
            out.append("")

        if problems:
            out.append("## Not documented")
            out.append("")
            out.append("| File | Reason |")
            out.append("| --- | --- |")
            for source, reason in problems:
                out.append(f"| `{_escape(source)}` | {_escape(reason)} |")
            out.append("")

        out.append("---")
        out.append("")
        out.append("Finding kinds: "
                   + "; ".join(f"**{KIND_LABELS[k]}** — {KIND_MEANINGS[k]}"
                               for k in KIND_ORDER if k in KIND_LABELS))
        out.append("")
        return "\n".join(out)

    def respond(self, openai_client: OpenAIClient, request_builder: RequestBuilder,
                chat: Chat = None, memories=None):
        _, output_root = roots_for(chat)
        try:
            index, specs, problems = self._load(output_root)
        except FileNotFoundError:
            return ("No specifications have been extracted yet. Tell the user to scan a "
                    "source directory first, and do not retry this.")
        except (OSError, json.JSONDecodeError) as e:
            return (f"The specification index could not be read ({type(e).__name__}). "
                    f"Tell the user and do not retry this.")

        if not specs:
            return ("The index lists no successfully documented files. Tell the user "
                    "nothing could be written.")

        document = self._render(index, specs, problems)
        target = os.path.join(output_root, DOCUMENT_NAME)
        try:
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(document)
        except OSError as e:
            return (f"The document could not be written ({type(e).__name__}). Tell the "
                    f"user and do not retry this.")

        methods = sum(len(s.get("methods") or []) for s in specs)
        questions = sum(len(m.get("uncertainties") or [])
                        for s in specs for m in (s.get("methods") or []))
        print(f"[AGENT] {self.name} wrote {DOCUMENT_NAME} ({len(document)} chars)")
        return (f"Wrote {DOCUMENT_NAME} in the spec output directory: {len(specs)} class(es), "
                f"{methods} method(s), {questions} open question(s)"
                + (f", {len(problems)} file(s) listed as not documented" if problems else "")
                + ". Tell the user where it is; do not repeat its contents.")
