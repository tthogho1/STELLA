# Custom agents

Agents that are not part of STELLA's demo set. Two unrelated things live here: a travel
search backed by Databricks, and a pipeline that documents source code with a local model.

Both are dropped into this directory and picked up at startup — `AgentStorage` walks
`stella_agents/` and instantiates every `Agent` subclass it finds, so nothing needs
registering. Add one to a workspace with `/add <agent id>`; `/agents` lists what is
available.

---

## Documenting source code

Three agents, run in order. Each is a leaf: they do their own work rather than delegating.

| Agent id | What it does |
| --- | --- |
| `method_spec_agent` | Reads **one file** and writes a specification of the methods it declares |
| `repository_spec_agent` | Walks a **directory**, documents every file in it, and writes an index |
| `spec_document_agent` | Turns those specifications into a **Markdown document** |

```
/add repository_spec_agent
/add spec_document_agent

> Document everything under com/example.
Scanned com/example: documented 3 file(s), 4 method(s), 6 open question(s).

> Now write the document.
Wrote specification.md in the spec output directory: 3 class(es), 4 method(s), ...
```

Output lands under `SPEC_OUTPUT_ROOT`, mirroring the source tree:

```
spec_output/
├── com/example/billing/BillingService.java.spec.json
├── com/example/customer/CustomerService.java.spec.json
├── index.json              what was covered, and what failed
└── specification.md        the document
```

### Getting the code onto the server

These agents read the server's disk. On a server that is not the machine holding the
code, upload a zip for the workspace first (`/upload`, see the main README) — the
workspace then reads that archive **instead of** `SPEC_SOURCE_ROOT`, and writes to
`SPEC_OUTPUT_ROOT/workspaces/<workspace id>/` rather than the shared output root.

Two workspaces analysing different code therefore cannot read each other's source or
overwrite each other's `index.json`. A workspace with no upload keeps the configured
roots, which is what a server running alongside the code wants.

The archive is unpacked as it is, so a zip made from a project directory carries its
wrapper: `/upload orders.zip` means naming `orders-main/com/example`, not `com/example`.
`/source` prints the top level after an upload for exactly this reason. Stripping the
wrapper automatically was tried and reverted — a single top-level directory is also what
an archive of one package looks like, and guessing wrong breaks the path the user names.

`/download` brings the generated specifications back the other way. It is the only way
to read them off a remote server: what an agent returns into the chat is a **digest**,
not the specification, because memories are re-sent on every later agent call and a whole
specification there would be truncated and would crowd out everything else.

### What a specification contains

Every finding carries the line it came from **and that line's source**, so a claim can be
checked without opening the file:

```json
{ "kind": "db_operation", "detail": "saves the order",
  "line": 20, "source_line": "orderRepo.save(order);" }
```

Findings are one of `db_operation`, `external_call`, `exception`, `transaction`, `branch`
or `other`. Anything unrecognised is `other` rather than guessed at.

Each method also carries `uncertainties`: questions a reviewer has to answer, because
nothing in the code settled them. The generated document collects these into a section of
their own. **A specification extracted from code describes what the code does, not what it
is supposed to do** — the two are not the same, and the open questions are where the
difference shows up.

### Open questions, and what they are not

The section at the end of the document is for what **no code anywhere settles**: intent,
a business rule, what should happen in a case the code does not handle.

It did not start out that way. The instruction was "anything you cannot determine from
this file alone goes in uncertainties" — and a call into another class is, by definition,
not determinable from this file. The model dutifully filed one question per collaborator.
Measured over three runs of the sample sources:

| | Questions | A person actually had to answer |
| --- | ---: | ---: |
| "cannot determine from this file alone" | 16 (5.3 per run) | 5 (31%) |
| Saying what an uncertainty *is* | 9 (3.0 per run) | 7 (78%) |

An independent six-run sample of the same wording produced 12, of which 11 survived
cleaning, and held the ratio (about 73%). Read all of these as the rough figures they
are: one three-run sample of the new wording came back with **nothing at all**, and six
runs went `1, 0, 3, 4, 1, 3`. The variance between runs is larger than the difference
this table is measuring, so three runs is not enough to conclude anything smaller than
the change above.

The rest were "how does `PaymentGateway.charge` work" — answerable by opening the file.
Telling the model that most methods raise none, and to return an empty list rather than
fill it, is part of what did it: one run of three now returns nothing at all.

Three limits hold whatever comes back: list markers a model wrote into the items are
stripped (`* how the validator instance is obtained` arrives exactly like that),
near-duplicates are collapsed (the same doubt is raised again every time a collaborator
appears), and `SPEC_MAX_QUESTIONS_PER_METHOD` caps the rest.

A question naming a class the scan documented is **annotated**, not dropped — the last
column points at the section that answers it. Dropping was tried against real output and
rejected: `what should happen if customerService.register(request) returns an exception`
names a documented class exactly as `what the customer service class entails` does, and
no rule separated them without taking the real question too. Losing one is worse than
carrying one that turns out to be answerable, so the reader is told where to look.

### What the model is and is not asked to do

Testing against `llama3.1:8b` and `qwen2.5-coder:7b` found a sharp split, and the design
follows it:

| | Reliable? | So it is |
| --- | --- | --- |
| **Where** something happens | Yes — 17 of 17 cited lines really showed what was claimed | asked of the model |
| **What** it is | No — one run filed a payment gateway call and a database save both as "exception" | read from the source line |
| The method **signature** | No — one run returned outputs `["Receipt", "T"]` | parsed from the declaration |

So the model reports a line number and a sentence; the kind and the signature come from
the code. That is why the classification is stable run to run while the prose is not.

Two more consequences of the same testing, both in the code:

- **One call per method.** Asking a single call for the signature *and* the findings made
  an 8B model drop the findings almost entirely — `db_operation` came back empty and only
  two of six kinds were used. Splitting them fixed it.
- **The whole file is in the prompt**, so a method's analysis picks up lines from its
  neighbour. Each method has a span, and findings outside it are discarded.

### Settings

Read from `app/.env`. See `app/.env_template` for the full list.

| Setting | Default | Notes |
| --- | --- | --- |
| `SPEC_SOURCE_ROOT` | `.` | **Only files under here are read.** The path arrives from a chat message by way of an LLM, so this is what stops it reading anything the process can |
| `SPEC_OUTPUT_ROOT` | `spec_output` | Where specifications and the document are written |
| `SPEC_SOURCE_SUFFIXES` | `.java` | Comma-separated file types the directory scan reads |
| `SPEC_MAX_FILES` | `50` | Files one scan documents before it stops |
| `SPEC_CONCURRENCY` | `4` | Files documented at once; keep under `OPENAI_MAX_WORKERS` |
| `SPEC_MAX_SOURCE_CHARS` | `40000` | A larger file is refused rather than truncated |
| `SPEC_MAX_METHODS_PER_FILE` | `12` | One model call per method, so a large class is a long request |
| `SPEC_MAX_QUESTIONS_PER_METHOD` | `3` | Open questions kept per method, after near-duplicates are collapsed |
| `SPEC_DB_PATTERN` | *(blank)* | Receivers counted as database access. Blank uses the default (`repo`, `dao`, `mapper`, `entitymanager`, `jdbc`, `session`) |
| `SPEC_EXTERNAL_PATTERN` | *(blank)* | Receivers counted as leaving the system (`gateway`, `client`, `api`, `kafka`, …) |
| `SPEC_DOCUMENT_NAME` | `specification.md` | The rendered document |
| `SOURCE_UPLOAD_ROOT` | `uploads` | Where `/upload` unpacks an archive, one directory per workspace. Must not overlap a directory agents are imported from; uploads are refused if it does |
| `SPEC_DOWNLOAD_MAX_BYTES` | `52428800` | Ceiling on what `/download` will zip up and send back |

The two pattern settings exist for a codebase with its own naming conventions. A blank or
unparseable value falls back to the default — an empty regular expression matches every
line, which would file a whole file as database access.

### Running it against a local model

Nothing leaves the machine. Point STELLA at any OpenAI-compatible endpoint:

```bash
OPENAI_BASE_URL="http://localhost:11434/v1"
OPENAI_API_KEY="ollama"
OPENAI_MODEL_ACTION_SELECTION="llama3.1:8b"
OPENAI_MODEL_RESPONSE="llama3.1:8b"
```

`llama3.1:8b` was the better of the two models tried, despite `qwen2.5-coder:7b` being the
code-specialised one: the job is classifying what a call means rather than writing code.
A model without tool-calling support needs `USE_TOOL_CALLING="false"`.

### Why the directory scan iterates instead of delegating

It walks the files itself rather than delegating one child task per file, which is not the
obvious shape for a tree framework. STELLA's action map holds one entry per agent id and
duplicate selections are collapsed, so a parent cannot fan out to the same agent once per
file. Nor should it: `os.walk` does not need a model, and routing each file through an
action-selection call would spend an LLM round apiece and hit the depth limit after a
handful.

The tree earns its place one level up, where different *kinds* of analysis run as
siblings — this pipeline alongside a database analyser, a test analyser, a reviewer.

---

## City search

`city_search_agent` — a travel destination search over a Wikivoyage index in Databricks
Vector Search, with an LLM-written summary. It wraps the LangChain tool in
`stella_agents/tools/city_search.py`, which is unchanged and still usable from LangChain.

Needs `pip install -e ".[city-search]"` and two credentials:

| Setting | Notes |
| --- | --- |
| `WORKSPACE_URL` | Databricks workspace URL, no trailing slash. Read per call |
| `DATABRICKS_TOKEN` | Personal access token. Read per call |
| `CITIES_INDEX_NAME`, `CITIES_ENDPOINT_NAME`, `CITIES_LLM_MODEL`, `CITIES_COLUMNS`, `CITIES_NUM_RESULTS` | Read at import — changing these needs a restart, not `/agent/reload` |

Without the credentials the agent still loads and every search reports that the lookup is
unavailable, so the server starts either way.

One thing to know about the index: it is a similarity search with no notion of exclusion.
Asked for "coffee outside Africa" it returns African cities and the summary says the
results are not relevant — which a coordinator reads as the request being unanswered, and
re-runs the identical search. The agent's result says outright that repeating the search
returns the same rows, which is what stops that loop.

---

## Writing another one

```python
# stella_agents/CustomAgents/MyAgent.py
from stella_core.models.agent import Agent


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id='my_agent',                        # unique; this is what /add takes
            name='MY_AGENT',                            # shown in progress updates
            short_description='What this agent does',   # the coordinator picks from this
            skip_action_selection=True,                 # a leaf: nothing to delegate to
            forward_all_memory_entries_to_parent=True,  # hand the result back up
        )

    def respond(self, openai_client, request_builder, chat=None, memories=None):
        return "the result, as a string"
```

Two constraints: subclass `Agent`, and be constructible with **no arguments**. Beyond
that, three things are worth copying from the agents here:

- **Make `short_description` say what the agent can do.** It is all the coordinator reads
  when deciding whether to delegate.
- **Say when not to retry.** A result that reads as empty makes the coordinator call the
  agent again; if there is nothing more to get, the result should say so.
- **Keep what you return short.** It goes into `memories`, which are re-sent on every
  later agent call. Write the detail to a file and return a digest and its path.
