"""
Documents every source file under a directory.

Why this iterates rather than delegating one child per file: STELLA's action map holds
one entry per agent id and duplicate selections are collapsed, so a parent cannot fan out
to the same agent once per file. Nor should it -- listing files is a job for os.walk, and
routing each one through an action-selection call would spend an LLM round per file and
run into the depth limit after a handful.

The tree still earns its place one level up, where different *kinds* of analysis run as
siblings: this agent alongside a database analyser, a test analyser, a reviewer.
"""
import concurrent.futures
import json
import os

from stella_core.models.agent import Agent
from stella_core.models.chat import Chat
from stella_core.openai_client import OpenAIClient
from stella_core.utils.request_builder import RequestBuilder

from stella_agents.CustomAgents.MethodSpecAgent import (
    MethodSpecAgent, SpecExtractionError, SPEC_OUTPUT_ROOT, SPEC_SOURCE_ROOT,
    MAX_SOURCE_CHARS, roots_for)

# Which files are worth reading. Everything else under the root is skipped silently.
SOURCE_SUFFIXES = tuple(
    s.strip() for s in os.getenv("SPEC_SOURCE_SUFFIXES", ".java").split(",") if s.strip())

# Directories never worth walking into.
SKIP_DIRS = {"target", "build", "out", "node_modules", ".git", ".idea", "__pycache__"}

# A scan is one model call per method, so a repository is a long job. Cap it rather than
# start something that runs for an hour.
MAX_FILES = int(os.getenv("SPEC_MAX_FILES", "50"))

# Files documented at once. Each holds an OpenAI worker for the length of its analysis,
# so this should stay under OPENAI_MAX_WORKERS.
CONCURRENCY = max(1, int(os.getenv("SPEC_CONCURRENCY", "4")))

INDEX_NAME = "index.json"


class RepositorySpecAgent(Agent):
    """Walks a source tree and writes a specification for each file in it."""

    def __init__(self):
        super().__init__(
            agent_id='repository_spec_agent',
            name='REPO_SPEC',
            short_description='Document every source file under a directory, writing one '
                              'specification per file plus an index of them',
            skip_action_selection=True,
            forward_all_memory_entries_to_parent=True,
        )

    def _find_directory(self, openai_client, chat, memories):
        """Pulls the directory to scan out of the conversation, defaulting to the root."""
        messages = [
            {"role": "system",
             "content": "Reply with the directory path mentioned in the conversation and "
                        "nothing else. If none is named, reply with a single dot."},
            {"role": "user",
             "content": f"{self._construct_chat_string(chat) if chat else ''}"
                        f"{self._construct_memory_string(memories) if memories else ''}"},
        ]
        answer = (openai_client.chat_completion(
            messages=messages, model=self.model_for_action_selection) or "").strip()
        return answer.strip('"\'`') or "."

    @staticmethod
    def _collect(directory, source_root=None):
        """
        Every source file under `directory`, relative to the source root.

        The directory is resolved through the same containment check as a single file, so
        a path from the conversation cannot walk out of the tree.
        """
        source_root = os.path.abspath(source_root or SPEC_SOURCE_ROOT)
        cleaned = directory.strip().strip('"\'`').lstrip("/")
        root = os.path.abspath(os.path.join(source_root, cleaned))
        if not (root == source_root or root.startswith(source_root + os.sep)):
            raise ValueError(f"{directory} is outside the source root")
        if not os.path.isdir(root):
            raise NotADirectoryError(f"{directory} is not a directory under the source root")

        found = []
        for current, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS]
            for name in sorted(files):
                if name.endswith(SOURCE_SUFFIXES):
                    found.append(os.path.relpath(os.path.join(current, name),
                                                 source_root))
        return sorted(found)

    def _document_one(self, openai_client, relative, source_root=None, output_root=None):
        """Reads and documents a single file. Never raises; failures become a record."""
        path = os.path.join(source_root or SPEC_SOURCE_ROOT, relative)
        try:
            source = open(path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            return {"source": relative, "error": f"could not be read ({type(e).__name__})"}

        if len(source) > MAX_SOURCE_CHARS:
            return {"source": relative,
                    "error": f"is {len(source)} characters, over the {MAX_SOURCE_CHARS} limit"}

        try:
            return MethodSpecAgent().document_source(openai_client, source, relative,
                                                     output_root)
        except SpecExtractionError as e:
            return {"source": relative, "error": str(e)}
        except Exception as e:                                   # noqa: BLE001
            print(f"[AGENT] REPO_SPEC {relative} failed ({type(e).__name__}: {e})")
            return {"source": relative, "error": f"failed ({type(e).__name__})"}

    @staticmethod
    def _write_index(directory, results, output_root=None):
        """An index of what was covered, so a later stage does not have to walk the tree."""
        root = output_root or SPEC_OUTPUT_ROOT
        target = os.path.join(root, INDEX_NAME)
        os.makedirs(root, exist_ok=True)
        document = {
            "scanned": directory,
            "files": [
                {"source": r["source"] if "error" in r else r["relative"],
                 "spec": r.get("written_to"),
                 "class_name": r.get("class_name"),
                 "methods": len(r.get("methods") or []),
                 "error": r.get("error")}
                for r in results
            ],
        }
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
        return INDEX_NAME

    @staticmethod
    def _digest(directory, results, index_name, skipped):
        done = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        methods = sum(len(r.get("methods") or []) for r in done)
        questions = sum(len(m.get("uncertainties") or [])
                        for r in done for m in (r.get("methods") or []))

        lines = [f"Scanned {directory}: documented {len(done)} file(s), {methods} method(s), "
                 f"{questions} open question(s)."]
        for result in done:
            lines.append(f"  - {result['relative']} ({result.get('class_name', '?')}, "
                         f"{len(result.get('methods') or [])} method(s))")
        for result in failed:
            lines.append(f"  ! {result['source']}: {result['error']}")
        if skipped:
            lines.append(f"  {skipped} further file(s) were not scanned (limit {MAX_FILES}).")
        lines.append(f"Each specification, with line numbers, is under the spec output "
                     f"directory; {index_name} lists them. Do not re-scan this directory.")
        return "\n".join(lines)

    def respond(self, openai_client: OpenAIClient, request_builder: RequestBuilder,
                chat: Chat = None, memories=None):
        source_root, output_root = roots_for(chat)
        directory = self._find_directory(openai_client, chat, memories)

        try:
            files = self._collect(directory, source_root)
        except (ValueError, NotADirectoryError, OSError) as e:
            return (f"Could not scan {directory}: {e}. Tell the user and do not retry the "
                    f"same path.")

        if not files:
            return (f"No files matching {', '.join(SOURCE_SUFFIXES)} were found under "
                    f"{directory}. Tell the user and do not retry.")

        skipped = max(0, len(files) - MAX_FILES)
        files = files[:MAX_FILES]
        print(f"[AGENT] {self.name} scanning {directory}: {len(files)} file(s), "
              f"{CONCURRENCY} at a time")

        # Threads, because each file is spent waiting on the model. The OpenAIClient
        # queue is what actually bounds how many requests are in flight.
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            results = list(pool.map(
                lambda relative: self._document_one(
                    openai_client, relative, source_root, output_root), files))

        try:
            index_name = self._write_index(directory, results, output_root)
        except OSError as e:
            print(f"[AGENT] {self.name} could not write the index: {e}")
            index_name = "(the index could not be written)"

        return self._digest(directory, results, index_name, skipped)
