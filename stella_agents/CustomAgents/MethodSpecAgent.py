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

KIND_MEANINGS = {
    "db_operation": "reads or writes persistent data: repository, DAO, JPA, mapper or raw SQL",
    "external_call": "leaves this system: HTTP, a gateway, a queue, another service",
    "exception": "an exception this method throws or lets propagate",
    "transaction": "transaction handling, such as an @Transactional annotation",
    "branch": "a conditional that changes what the method does",
}

# Two passes, not one. Asking a single call for the signature *and* the findings made an
# 8B model drop the findings almost entirely: with method_name/inputs/outputs in the same
# schema, db_operation came back empty and only two kinds were used at all; with the
# findings alone it found every repository call. Measured on llama3.1:8b, temperature 0,
# same file and prompt -- the only difference was the extra fields.
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
                    "kind": {
                        "type": "string",
                        "enum": list(KIND_MEANINGS),
                        "description": "; ".join(
                            f"{k}: {v}" for k, v in KIND_MEANINGS.items()),
                    },
                    "detail": {"type": "string"},
                    "line": {"type": "integer",
                             "description": "the line number shown in the input"},
                },
                "required": ["kind", "detail", "line"],
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
            return (f"The model's answer for {relative} was not the requested JSON. Tell "
                    f"the user the extraction failed.")
        except Exception as e:
            print(f"[AGENT] {self.name} outline failed ({type(e).__name__}: {e})")
            return (f"Could not read the methods of {relative} ({type(e).__name__}). Tell "
                    f"the user and do not retry it.")

        methods = self._declared_only(outline.get("methods") or [], source)
        if not methods:
            return f"No methods were found in {relative}. Tell the user."

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
            return (f"Documented {relative} but could not save it ({type(e).__name__}). "
                    f"Tell the user the specification was not written.")

        print(f"[AGENT] {self.name} wrote {written_to}")
        return self._digest(relative, class_name, methods, written_to)
