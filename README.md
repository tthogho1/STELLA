<div align="center">
  <img src="assets/images/STELLA_QUICK_DEMO.gif" alt="STELLA Banner">
</div>

<div align="center">

[![License](https://img.shields.io/badge/license-AGPLv3-blue.svg)](LICENSE) — [![Version](https://img.shields.io/badge/version-beta-orange.svg)](https://github.com/Norditech-AB/STELLA/tree/main) — [![Community](https://img.shields.io/badge/community-active-ff69b4.svg)](https://docs.stellaframework.com/Community.html) — [![Documentation](https://img.shields.io/badge/documentation-here-32a875.svg)](https://docs.stellaframework.com/)

</div>

# Welcome to STELLA

### A Scalable Multi-Agent AI Framework

STELLA is a multi-agent framework for conversational agents using Large Language Models that focuses on scalability, broad capabilities, and powerful configuration. It simplifies incorporation of advanced LLM capabilities into applications, offering a server-based multi-agent framework solution that is powerful, efficient and scalable.

**🛠️ Status**: Beta - We're crafting the future.
STELLA is currently in beta. We are working hard to improve the framework and add new features. For questions or feedback, contact us at [contact@stellaframework.com](mailto:contact@stellaframework.com).

## 📚 Guide

- [🚀 Getting Started](https://docs.stellaframework.com/Getting_Started.html)
- [📖 Documentation](https://docs.stellaframework.com/)
- [🤝 Contributing](https://docs.stellaframework.com/contribution_guidelines/index.html)
- [🌍 Community](https://docs.stellaframework.com/Community.html)
- [⚖️ Licensing](https://docs.stellaframework.com/Licensing.html)
- [💫 VISS.AI - Accessible AI for all](#vissai---accessible-ai-for-all) (**COMING SOON** 🚀)


### Quick Start

To set up STELLA:

1. **Clone the Repository**:

```bash
git clone https://github.com/Norditech-AB/STELLA.git
```
```bash
git clone git@github.com:Norditech-AB/STELLA.git
```
```bash
gh repo clone Norditech-AB/STELLA
```

2. **Create & Activate a Virtual Environment**:

```bash
python -m venv venv
```
```bash
source venv/bin/activate
```

3. **Install STELLA**: Navigate to the root directory and run:
```bash
pip install -e .
```

4. **Quickly Configure STELLA** by following the instructions after running:

```bash
stella configure
```

5. **Start the Server** by running
 
```bash
stella serve
```

6. **Explore and Create**: Open a new terminal window and run
```bash
stella
```
This will open a Python shell with the STELLA environment loaded. You can now explore the framework and create your own agents.
Get started quickly writing `/register`.
To list available commands, write `/help` in the shell.

<div align="center">
  <img src="assets/images/REGISTER_AND_TALK_TO_STELLA.gif" alt="STELLA Banner">
</div>

For a complete guide, visit [Getting Started](https://docs.stellaframework.com/Getting_Started).

> **Upgrading an existing checkout?** Run `pip install -e .` again after pulling. The
> runtime and the agents were split into their own top-level packages, and an editable
> install made before that will not see them:
> `ModuleNotFoundError: No module named 'stella_agents'`.

### Connecting to a Remote Server

The CLI is a client and nothing more -- it never imports the server -- so it can talk to
one on another machine. `cli/config.json`:

```json
{
  "host": "stella.example.com",
  "ssl": true
}
```

`"port"` may be left out when a reverse proxy answers on the standard port. `"ssl"`
defaults to `false`, which is what localhost wants.

**Turn `ssl` on for anything that is not localhost.** Over plain http the CLI sends the
password on `/login` and the JWT on every request after it in the clear. STELLA does not
terminate TLS itself, so put a reverse proxy in front of it.

For one or two people an SSH tunnel is less work than a certificate, and needs no
configuration change at all:

```bash
ssh -L 5001:localhost:5001 you@stella.example.com
```

### Creating an Agent

Registering an agent is two steps: STELLA has to **find** the class, and a workspace has
to **use** it.

**1. Drop a file into `stella_agents/`.** There is no registration list to edit —
`AgentStorage` walks the directory at startup, imports every `.py` file and instantiates
any `Agent` subclass it finds. A file that fails to import is skipped with a message
rather than stopping the server.

```python
# stella_agents/CustomAgents/MyAgent.py
from stella_core.models.agent import Agent


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            agent_id='my_agent',                        # unique, this is what /add takes
            name='MY_AGENT',                            # shown in progress updates
            short_description='What this agent does',   # the coordinator picks from this
            skip_action_selection=True,                 # a leaf: nothing to delegate to
            forward_all_memory_entries_to_parent=True,  # hand the result back up
        )

    def respond(self, openai_client, request_builder, chat=None, memories=None):
        return "the result, as a string"
```

Two rules: the class must subclass `Agent`, and it must be constructible with **no
arguments**. Make `short_description` say what the agent can do — the coordinator reads
only that when deciding whether to delegate to it.

Pick up the new file with `stella serve` (restart) or `GET /agent/reload`.

**2. Add it to a workspace** from the CLI. `/agents` lists everything the server has
loaded, marking what this workspace already has:

```
/agents
    brewery_agent               Fetch brewery data
  * demo_weather_agent          Fetch weather data
    my_agent                    What this agent does

/add my_agent
```

Until you do, the agent exists but nothing will call it. With no agents at all a
workspace answers through `stella_welcome_agent`; add one and the coordinator takes over
and can delegate to several agents at once.

### Running Agents Without the Server

`stella_core` imports no web framework, so a request can be run straight from Python --
no Flask, no SocketIO, no HTTP. Build what the server would build, create a task, then
keep calling `Task.execute()` with whatever it returns. The queue in `TaskManager` is a
thread pool around exactly that loop.

```bash
cd app && python ../examples/run_without_a_server.py "What is the weather in Kyoto?"
```

```
   [*] Asking WEATHER…
   [*] WEATHER done
   [*] Writing the answer…

A: The current weather in Kyoto shows a temperature of 28.2°C ...

stella_coordinator_agent 2.5s x2  — answered the user
└─ demo_weather_agent 1.8s

2 tasks, 3 agent runs, 4.306s wall clock
```

See [`examples/run_without_a_server.py`](examples/run_without_a_server.py). Run it from
`app/`, for the same reason `stella serve` chdirs there: `.env` and the SQLite path are
resolved against the working directory.

Two things the server does that a library caller has to do itself, because neither
happens as a side effect of an import any more:

- `init_database()` — picks the backend from `DATABASE`
- `configure_default_agents(...)` — the runtime has no built-in coordinator id

Progress and answers go to an `EventSink` instead of SocketIO. `CollectingSink` keeps
them in memory; see `stella_core/events.py`.

### Seeing What a Request Did

Only the final answer reaches the user, so `/trace` in the CLI (or
`GET /chat/trace?chat_id=...`) reconstructs the run afterwards: which agents were
delegated to, in what shape, how many times each ran, and where the time went. Siblings
appear in the order they were delegated, and the total is wall clock rather than the sum
of the spans, since parallel agents overlap.

### Repository Layout

| Directory | Contents |
| --- | --- |
| `stella_core/` | The agent runtime — `Task`, `Agent`, `AgentStorage`, the queues, the database layer, `OpenAIClient`, `EventSink`. Imports no web framework, so it can be driven without a server. |
| `stella_agents/` | The agents that ship with STELLA, and where your own belong. See [`CustomAgents/README.md`](stella_agents/CustomAgents/README.md) for the ones beyond the demo set, including a pipeline that documents source code with a local model. |
| `app/` | The reference Flask + SocketIO server that wires the runtime together, plus thin re-exports so older agents importing from `app.*` keep working. |
| `cli/` | The client. Talks to the server over HTTP and SocketIO; it never imports the server. |

`app/agents/` is separate from `stella_agents/`: it is where `GET /agent/download`
installs community packages at runtime, kept apart so downloaded content never overwrites
what the repository ships.

## Typical Use Cases:

- Automating workflows and tasks.
- Building digital workforces.
- Connecting APIs and services.
- Creating smart chatbots.
- Speeding up the development of LLM-powered applications.
- Finding patterns and trends in unstructured data.
- Creating content.

**Get started with our [Getting Started Guide](https://docs.stellaframework.com/Getting_Started).**

## Key Features

- **Server-based Architecture**: Solid foundation for large-scale applications.
- **Socket Communication**: Efficient real-time updates.
- **Multi-User Support**: Scalable for numerous users.
- **Configurability**: Expandable with minimal coding.
- **Speed Optimization**: Faster execution times.
- **Agent Intercommunication**: Reduces Token usage.
- **Single-Prompt Method**: Simple model support.
- **Command Line Interface**: Direct terminal access.
- **Community-Driven Package Manager**: Easy publication and installation of agents.

For detailed information, visit our [Documentation page](https://docs.stellaframework.com/).

## LICENSE
STELLA offers two licensing options to accommodate diverse use cases. Both licenses are designed to meet different requirements, ensuring flexibility and convenience for all users.
- **AGPL-3.0 License:** This license is perfect for students and hobbyists. It's an [Open Source Initiative (OSI)-approved](https://opensource.org/licenses/) open-source license that encourages collaborative development and knowledge exchange. For detailed information, please refer to the LICENSE file.
- **Enterprise License:** Tailored for commercial purposes, this license allows for the integration of STELLA software into commercial products and services. This option is ideal for those who need to use our solutions in a commercial context without adhering to the open-source stipulations of the AGPL-3.0 license. For inquiries and more details about this license, please [contact us](mailto:philip@norditech.se).

## Community, Support & Contributions

Join our community for support, discussions, and insights. Connect through our forums and support channels.

Visit the [Community section](https://docs.stellaframework.com/Community) to get involved and see the [Contribution Guidelines](https://docs.stellaframework.com/contribution_guidelines/index.html) to learn how to contribute to STELLA.

## Next Steps

- Learn about agent creation in [Creating a new Agent](https://docs.stellaframework.com/agents/Creating_a_new_Agent).
- Explore CLI commands in the [CLI section](https://docs.stellaframework.com/cli/index).

---

STELLA is an evolving framework. We welcome contributions and feedback to improve and expand its capabilities.

---

# VISS.AI - Accessible AI for all

<div align="center">
  <img src="assets/images/JOIN_VISS_AI.png" alt="VISS.AI Waitlist Banner">
</div>

## Create AI-powered applications without writing a single line of code.
VISS.AI is a visual tool that allows you to create AI-powered applications without writing a single line of code. Integrate automatically towards APIs and services and create your own AI-powered applications in minutes.

Join the [🔗 VISS.AI Waitlist](https://viss.ai) to be the first to know when VISS.AI launches. Early access is being rolled out in batches.

<div align="center">
  <img src="assets/images/VISS_AI_FEATURES.png" alt="VISS.AI Features - Automatic integrations, accessible for everyone, instant deployment">
</div>

## How does it work?
- 🔗 **Simply connect your apps by clicking and dragging.** No coding required.
- 🗣️ Tell VISS.AI how your apps should interact with each other by describing your workflow.
- 🌍 Integrate automatically towards APIs and services.
- 🌟 Don't worry about data mapping, VISS.AI does it for you.
- 👨‍💻👩‍💻 Create your own AI-powered applications in minutes.
- 🚀 Deploy small or on scale with a single click.

<div align="center">
  <img src="assets/images/GENIUS_HEADER_VISS_AI_STELLA.png" alt="Does it really take a genius to work with AI?">
</div>

**Imagine a world where everyone can access the power of AI. VISS.AI is a STELLA-powered platform that makes this possible.**

Interesting? Watch the [🔗 VISS.AI Demo](https://www.youtube.com/watch?v=_c6GEbI1bjU) and join [🔗 the waitlist](https://viss.ai) to be the first to know when VISS.AI launches.