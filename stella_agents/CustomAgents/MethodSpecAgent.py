"""
Extracts a specification from one source file, with a line number behind every claim.

A specification nobody can check against the code is worse than none, so every finding
carries the line it came from. Testing showed both llama3.1:8b and qwen2.5-coder:7b cite
lines accurately (17 of 17 citations landed on a line that really shows what was claimed)
but disagree on what a finding *is*: qwen filed every `orderRepo.*` call as an external
call rather than database access. The kind descriptions below are written into the schema
for that reason -- the model is told what each value means rather than left to guess.
"""
import json
import os
import re

from stella_core.models.agent import Agent
from stella_core.models.chat import Chat
from stella_core.openai_client import OpenAIClient
from stella_core.utils.request_builder import RequestBuilder

# Only files under here may be read. The path arrives from a chat message by way of an
# LLM, so without a root this agent would read anything the process can.
SPEC_SOURCE_ROOT = os.path.abspath(os.getenv("SPEC_SOURCE_ROOT", "."))

# One source file per call. A large file costs context in every later agent call too,
# because the result travels up through memories.
MAX_SOURCE_CHARS = int(os.getenv("SPEC_MAX_SOURCE_CHARS", "40000"))

# One model call per method, so a large class is a long request. Cap it rather than
# let one file spend an unbounded amount of time.
MAX_METHODS_PER_FILE = int(os.getenv("SPEC_MAX_METHODS_PER_FILE", "12"))

# Where the full specification is written. The result travels up through memories, which
# are re-sent on every later agent call and capped by MAX_MEMORY_ENTRY_CHARS -- a whole
# specification there would be truncated and would crowd out everything else. The file
# holds the structured form for whatever builds the document; memories get a digest and
# the path to it.
SPEC_OUTPUT_ROOT = os.path.abspath(os.getenv("SPEC_OUTPUT_ROOT", "spec_output"))

# How many methods the digest names before it stops. The file always has all of them.
MAX_METHODS_IN_DIGEST = int(os.getenv("SPEC_MAX_METHODS_IN_DIGEST", "8"))

# How far from the cited line to look for the declaration. Asked for the signature line, a
# model cites the annotation above it often enough that an exact match loses real methods.
LINE_TOLERANCE = 3

# What a finding is, decided from the source line rather than by the model.
#
# Line citations came back accurate every time (17 of 17), but the *kind* did not: the
# same file, model and prompt classified all nine of one method's findings as "exception"
# on one run -- payment and a database save included -- and correctly on another. The
# model is reliable at saying where something happens and unreliable at saying what it is,
# so it is asked only for the former and the latter is read off the code.
def _pattern(name, default):
    """
    Compiles an overridable pattern, treating a blank setting as unset.

    An empty regex matches every line, so `SPEC_DB_PATTERN=""` -- which is exactly what a
    key shipped blank in .env_template produces -- would file the whole file as database
    access. A bad pattern falls back rather than stopping the server: this runs at import,
    and an agent that cannot be imported takes the rest of the scan down with it.
    """
    value = (os.getenv(name) or "").strip()
    if not value:
        return re.compile(default, re.IGNORECASE)
    try:
        return re.compile(value, re.IGNORECASE)
    except re.error as e:
        print(f"[AGENT] METHOD_SPEC {name} is not a valid regular expression ({e}); "
              f"using the default")
        return re.compile(default, re.IGNORECASE)


DB_RECEIVERS = _pattern(
    "SPEC_DB_PATTERN",
    r"\b\w*(repo|repository|dao|mapper|entitymanager|jdbc|session)\w*\s*\.")
EXTERNAL_RECEIVERS = _pattern(
    "SPEC_EXTERNAL_PATTERN",
    r"\b\w*(gateway|client|api|http|rest|feign|kafka|queue|sqs|sns|mail|smtp|"
    r"webhook|publisher)\w*\s*\.")
SQL_STATEMENT = re.compile(r"\b(select|insert into|update\s+\w+\s+set|delete from|merge into)\b",
                           re.IGNORECASE)
THROWS = re.compile(r"\bthrow\b|orElseThrow|\bcatch\s*\(")
TRANSACTIONAL = re.compile(r"@Transactional|beginTransaction|\.commit\(|\.rollback\(")
DECLARATION = re.compile(
    r"^\s*(public|protected|private|static|final|abstract|synchronized|\s)*"
    r"[\w<>\[\],\s]+\s+\w+\s*\([^;]*$")
SIGNATURE = re.compile(
    r"^(?:(?:public|protected|private|static|final|abstract|synchronized|default)\s+)*"
    r"(?:<[^>]+>\s+)?"                     # a generic declaration, discarded
    r"(?P<returns>[\w.<>\[\],\s]*?)\s*"    # empty for a constructor
    r"(?P<name>\w+)\s*\((?P<params>[^)]*)\)")
BRANCHES = re.compile(r"^\s*(if|else\s+if|switch|case|while|for)\b|\?.*:")

KIND_MEANINGS = {
    "db_operation": "reads or writes persistent data",
    "external_call": "leaves this system",
    "exception": "throws or propagates an exception",
    "transaction": "transaction handling",
    "branch": "a conditional that changes what the method does",
    "other": "a step that is none of the above",
}


def classify(line_text):
    """
    Decides what a line is, from the line itself.

    Order matters: `throw new OrderLockedException(id)` inside an if-branch is an
    exception first, and a repository call inside a transactional method is still a
    database operation. Anything unrecognised is "other" rather than guessed at.
    """
    if THROWS.search(line_text):
        return "exception"
    if TRANSACTIONAL.search(line_text):
        return "transaction"
    if DB_RECEIVERS.search(line_text) or SQL_STATEMENT.search(line_text):
        return "db_operation"
    if EXTERNAL_RECEIVERS.search(line_text):
        return "external_call"
    if BRANCHES.search(line_text):
        return "branch"
    return "other"


OUTLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "class_name": {"type": "string"},
        "methods": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "method_name": {"type": "string"},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "line": {"type": "integer",
                             "description": "the line the method is declared on"},
                },
                "required": ["method_name", "inputs", "outputs", "line"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["class_name", "methods"],
    "additionalProperties": False,
}

FINDINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "responsibility": {"type": "string",
                           "description": "one sentence, from what the code does"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # No "kind" here on purpose -- see classify() above.
                    "detail": {"type": "string"},
                    "line": {"type": "integer",
                             "description": "the line number shown in the input"},
                },
                "required": ["detail", "line"],
                "additionalProperties": False,
            },
        },
        "uncertainties": {
            "type": "array",
            "items": {"type": "string"},
            "description": "questions a reviewer must answer, not code fragments",
        },
    },
    "required": ["responsibility", "findings", "uncertainties"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You document existing source code. Describe only what the code shows: never state a "
    "business rule it does not prove. Every finding cites the line number it came from, "
    "exactly as numbered in the input. Anything you cannot determine from this file alone "
    "goes in uncertainties, written as a question a reviewer should answer."
)


class SpecExtractionError(Exception):
    """A file could not be documented. The message says why, for the user."""


class MethodSpecAgent(Agent):
    """Turns one source file into a structured specification."""

    def __init__(self):
        super().__init__(
            agent_id='method_spec_agent',
            name='METHOD_SPEC',
            short_description='Read a source file and extract a specification of its methods, '
                              'with the line number behind every finding',
            skip_action_selection=True,
            forward_all_memory_entries_to_parent=True,
        )

    def _resolve(self, path):
        """
        Turns a path from the conversation into a file inside SPEC_SOURCE_ROOT.

        Rejects anything outside it -- the path is chosen by a model reading user text,
        so "../../etc/passwd" is a thing that can arrive here.
        """
        # Models like to answer with a leading slash ("/src/com/...") even when asked for
        # the path as written. Treat what comes back as relative to the root either way;
        # a genuine escape attempt is still caught by the containment check below.
        cleaned = path.strip().strip('"\'`').lstrip("/")
        candidate = os.path.abspath(os.path.join(SPEC_SOURCE_ROOT, cleaned))
        if not (candidate == SPEC_SOURCE_ROOT
                or candidate.startswith(SPEC_SOURCE_ROOT + os.sep)):
            raise ValueError(f"{path} is outside the source root")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"{path} is not a file under the source root")
        return candidate

    def _find_path(self, openai_client, chat, memories):
        """Pulls the file to analyse out of the conversation."""
        messages = [
            {"role": "system",
             "content": "Reply with the source file path mentioned in the conversation, "
                        "and nothing else. No quotes, no explanation."},
            {"role": "user",
             "content": f"{self._construct_chat_string(chat) if chat else ''}"
                        f"{self._construct_memory_string(memories) if memories else ''}"},
        ]
        return (openai_client.chat_completion(
            messages=messages, model=self.model_for_action_selection) or "").strip()

    @staticmethod
    def _number(source):
        return "\n".join(f"{i:>4}: {line}" for i, line in enumerate(source.splitlines(), 1))

    @staticmethod
    def _signature_from_source(declaration_line):
        """
        Reads the return type and parameters off the declaration.

        Same reasoning as classify(): the model is asked where the method is, and the
        signature is then read from the code. Left to the model it drifts -- one run
        returned outputs ["Receipt", "T"] and inputs ["orderId", "force", "String",
        "boolean"], mixing names, types and a stray generic into the same list.
        :return: (inputs, outputs), or (None, None) when the line cannot be parsed
        """
        match = SIGNATURE.match(declaration_line.strip())
        if not match:
            return None, None

        returns = (match.group("returns") or "").strip()
        outputs = [] if returns in ("", "void") else [returns]

        params, depth, current = [], 0, ""
        for char in match.group("params"):
            # split on commas that are not inside Map<K, V>
            if char in "<([":
                depth += 1
            elif char in ">)]":
                depth -= 1
            if char == "," and depth == 0:
                params.append(current)
                current = ""
            else:
                current += char
        params.append(current)

        inputs = []
        for param in params:
            words = param.replace("final ", "").strip().split()
            if len(words) >= 2:
                inputs.append(f"{words[-1]}: {' '.join(words[:-1])}")
            elif words:
                inputs.append(words[0])
        return inputs, outputs

    @staticmethod
    def _clean_findings(findings, source_lines, span):
        """
        Classifies each finding and drops the ones that are not steps of this method.

        Three things come back that should not be kept: the same line reported twice, the
        method's own signature line, and lines belonging to the next method along -- the
        whole file is in the prompt, so a neighbour's body is right there to be picked up.
        """
        start, end = span
        kept, seen = [], set()
        for finding in findings:
            line = finding.get("line")
            if not isinstance(line, int) or not (1 <= line <= len(source_lines)):
                continue
            if not (start <= line <= end):
                continue
            text = source_lines[line - 1]
            stripped = text.strip()
            # the signature itself is not a step in the method
            if DECLARATION.match(stripped):
                continue
            if line in seen:
                continue
            seen.add(line)
            finding["kind"] = classify(text)
            finding["source_line"] = stripped
            kept.append(finding)
        return sorted(kept, key=lambda f: f["line"])

    @staticmethod
    def _clean_outputs(outputs):
        """
        Normalises "no return value" to an empty list.

        Models write it as "void", "None", "No output" and an empty string, all of which
        would otherwise be rendered as if they were a type.
        """
        if not outputs:
            return []
        return [o for o in outputs
                if o and o.strip().lower() not in {"void", "none", "no output", "n/a", "-"}]

    @staticmethod
    def _write(relative_source, class_name, methods):
        """
        Writes the full specification next to a mirror of the source tree.

        The output path is derived from the source path, which _resolve already confined
        to SPEC_SOURCE_ROOT, so it cannot point outside the output root either.
        :return: the path written, relative to SPEC_OUTPUT_ROOT
        """
        target = os.path.join(SPEC_OUTPUT_ROOT, relative_source + ".spec.json")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        document = {
            "source": relative_source,
            "class_name": class_name,
            "methods": methods,
        }
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
        return os.path.relpath(target, SPEC_OUTPUT_ROOT)

    @staticmethod
    def _digest(relative_source, class_name, methods, written_to):
        """
        The short form that goes up into memories.

        Names what was covered and where the detail is, without carrying the findings
        themselves -- a coordinator deciding what to do next needs to know the file was
        handled, not what line 20 does.
        """
        total_findings = sum(len(m.get("findings") or []) for m in methods)
        open_questions = sum(len(m.get("uncertainties") or []) for m in methods)

        lines = [f"Documented {relative_source} (class {class_name}): "
                 f"{len(methods)} method(s), {total_findings} finding(s), "
                 f"{open_questions} open question(s)."]
        for method in methods[:MAX_METHODS_IN_DIGEST]:
            kinds = sorted({f["kind"] for f in (method.get("findings") or [])})
            lines.append(f"  - {method['method_name']}"
                         f"({', '.join(method.get('inputs') or [])})"
                         f" -> {', '.join([o for o in (method.get('outputs') or []) if o]) or 'void'}"
                         + (f"  [{', '.join(kinds)}]" if kinds else ""))
        if len(methods) > MAX_METHODS_IN_DIGEST:
            lines.append(f"  - ... and {len(methods) - MAX_METHODS_IN_DIGEST} more")
        lines.append(f"The full specification with line numbers is in {written_to} "
                     f"under the spec output directory. Do not re-analyse this file.")
        return "\n".join(lines)

    @staticmethod
    def _declared_only(methods, source):
        """
        Drops entries whose cited line is not a declaration, and corrects the line if it
        is close.

        The outline pass returns methods the class calls as well as those it declares, and
        the cited line tells them apart: a declaration names the method followed by its
        parameter list, a call has a receiver in front of it. The line is allowed to be a
        couple off, because a model asked for the signature line will happily cite the
        @Transactional annotation above it instead.
        """
        lines = source.splitlines()
        kept = []
        for method in methods:
            line = method.get("line")
            name = method.get("method_name") or ""
            if not name or not isinstance(line, int):
                continue

            found = None
            for candidate in range(max(1, line - LINE_TOLERANCE),
                                   min(len(lines), line + LINE_TOLERANCE) + 1):
                text = lines[candidate - 1]
                if f"{name}(" in text and f".{name}(" not in text:
                    found = candidate
                    break
            if found is None:
                print(f"[AGENT] METHOD_SPEC dropping {name!r}: no declaration near L{line}")
                continue
            if found != line:
                print(f"[AGENT] METHOD_SPEC {name!r} declared on L{found}, not L{line}")
            method["line"] = found
            kept.append(method)
        return kept

    def _ask(self, openai_client, numbered, instruction, schema, name):
        """One structured-output call against the numbered source."""
        reply = openai_client.chat_completion(
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"{numbered}\n\n{instruction}"}],
            model=self.model_for_response,
            response_format={"type": "json_schema",
                             "json_schema": {"name": name, "strict": True, "schema": schema}},
        )
        return json.loads(reply)

    def document_source(self, openai_client, source, relative):
        """
        Turns one file's text into a written specification.

        Separate from respond() so the repository agent can reuse it: STELLA cannot
        delegate to the same agent once per file (the action map holds one entry per
        agent id and duplicate selections are collapsed), and picking files is not a
        job that needs a model anyway.
        :raises SpecExtractionError: with a reason the caller can put in a message
        :return: {"relative", "class_name", "methods", "written_to"}
        """
        numbered = self._number(source)
        print(f"[AGENT] {self.name} documenting {relative} ({len(source)} chars)")

        try:
            # "List every method" got the methods this class *calls* -- charge, save,
            # build -- alongside the ones it declares. The distinction has to be spelled
            # out, and the declaration line is what makes it checkable afterwards.
            outline = self._ask(
                openai_client, numbered,
                "List only the methods DECLARED in this class: the ones whose body appears "
                "in this file, each with the line its signature is on. Do not list methods "
                "that are merely called from inside those bodies.",
                OUTLINE_SCHEMA, "Outline")
        except json.JSONDecodeError:
            raise SpecExtractionError("the model's answer was not the requested JSON")
        except Exception as e:
            print(f"[AGENT] {self.name} outline failed ({type(e).__name__}: {e})")
            raise SpecExtractionError(f"the method list could not be read ({type(e).__name__})")

        methods = self._declared_only(outline.get("methods") or [], source)
        if not methods:
            raise SpecExtractionError("no methods were found")

        # Where each method's body ends: the line before the next declaration. Findings
        # outside that span belong to a different method -- with the whole file in the
        # prompt, one method's analysis picks up lines from its neighbour.
        # Sorted by declaration line first: the outline returns methods in whatever order
        # the model listed them, and taking "the next one" from an unsorted list produces
        # a span that ends before it starts, which silently discards every finding.
        methods.sort(key=lambda m: m["line"])
        spans = {}
        starts = [m["line"] for m in methods]
        last_line = len(source.splitlines())
        for index, method in enumerate(methods):
            end = starts[index + 1] - 1 if index + 1 < len(starts) else last_line
            # start a few lines early: an annotation above the signature belongs to it
            start = max(1, method["line"] - LINE_TOLERANCE)
            spans[method["method_name"]] = (start, max(start, end))

        # One call per method. Sharing a call between methods costs the same accuracy the
        # signature fields did: findings start belonging to whichever method came last.
        for method in methods[:MAX_METHODS_PER_FILE]:
            try:
                detail = self._ask(
                    openai_client, numbered,
                    f"Document ONLY the method `{method['method_name']}`, declared on line "
                    f"{method.get('line')}. Ignore every other method in the file.",
                    FINDINGS_SCHEMA, "Findings")
            except Exception as e:
                print(f"[AGENT] {self.name} findings failed for {method['method_name']} "
                      f"({type(e).__name__}: {e})")
                method["uncertainties"] = [f"This method could not be analysed "
                                           f"({type(e).__name__}); it needs a manual review."]
                continue
            method.update(detail)
            method["findings"] = self._clean_findings(
                method.get("findings") or [], source.splitlines(),
                span=spans[method["method_name"]])
            # Prefer the signature the code actually declares over the model's reading.
            declared_inputs, declared_outputs = self._signature_from_source(
                source.splitlines()[method["line"] - 1])
            if declared_inputs is not None:
                method["inputs"] = declared_inputs
                method["outputs"] = declared_outputs
            else:
                method["outputs"] = self._clean_outputs(method.get("outputs"))

        if len(methods) > MAX_METHODS_PER_FILE:
            methods = methods[:MAX_METHODS_PER_FILE]
            methods.append({"method_name": f"({len(outline['methods']) - MAX_METHODS_PER_FILE} "
                                           f"further methods not analysed)",
                            "inputs": [], "outputs": [], "findings": [], "uncertainties": []})

        class_name = outline.get("class_name", "?")
        try:
            written_to = self._write(relative, class_name, methods)
        except OSError as e:
            print(f"[AGENT] {self.name} could not write the specification: {e}")
            raise SpecExtractionError(f"the specification could not be saved ({type(e).__name__})")

        print(f"[AGENT] {self.name} wrote {written_to}")
        return {"relative": relative, "class_name": class_name,
                "methods": methods, "written_to": written_to}

    def respond(self, openai_client: OpenAIClient, request_builder: RequestBuilder,
                chat: Chat = None, memories=None):
        raw_path = self._find_path(openai_client, chat, memories)
        if not raw_path:
            return "No source file was named. Ask the user which file to document."

        try:
            path = self._resolve(raw_path)
            source = open(path, encoding="utf-8", errors="replace").read()
        except (ValueError, FileNotFoundError, OSError) as e:
            return (f"Could not read {raw_path}: {e}. Tell the user and do not retry the "
                    f"same path.")

        if len(source) > MAX_SOURCE_CHARS:
            return (f"{raw_path} is {len(source)} characters, over the {MAX_SOURCE_CHARS} "
                    f"limit for one pass. Tell the user it has to be split, and do not "
                    f"retry it.")

        relative = os.path.relpath(path, SPEC_SOURCE_ROOT)
        try:
            result = self.document_source(openai_client, source, relative)
        except SpecExtractionError as e:
            return (f"Could not document {relative}: {e}. Tell the user and do not "
                    f"retry this file.")
        return self._digest(result["relative"], result["class_name"],
                            result["methods"], result["written_to"])
