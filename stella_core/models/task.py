import json
import os

from stella_core.db import db
from stella_core.models.chat import Chat
from stella_core.agent_storage import AgentStorage
from stella_core.openai_client import OpenAIClient
from stella_core.utils.request_builder import RequestBuilder

# Tell the user which agents are working while a request is being handled. Only the final
# answer is ever sent as a chat message, so without these the client sits silent for the
# whole tree. Set EMIT_PROGRESS_EVENTS="false" to go back to saying nothing.
EMIT_PROGRESS_EVENTS = os.getenv('EMIT_PROGRESS_EVENTS', 'true').strip().lower() not in ('false', '0', 'no')

# The agent ids create_top_level_task() falls back to when a workspace has no
# coordinator_agent of its own. Unset by default: stella_core has no built-in notion of
# "the" coordinator or welcome agent, so a workspace with nothing configured and no
# default set here simply fails the request (caught and reported like any other agent
# failure) rather than reaching for a STELLA-specific agent id that may not even be
# loaded. Call configure_default_agents() once at startup to set them.
_default_general_agent_id = None
_default_empty_workspace_agent_id = None


def configure_default_agents(general_agent_id=None, empty_workspace_agent_id=None):
    """
    Sets the fallback agent ids used when a workspace has no coordinator_agent of its own.
    :param general_agent_id: Used when the workspace has agents connected but no
                             coordinator of its own.
    :param empty_workspace_agent_id: Used when the workspace has no agents connected at
                                     all (e.g. to help the user get started).
    """
    global _default_general_agent_id, _default_empty_workspace_agent_id
    _default_general_agent_id = general_agent_id
    _default_empty_workspace_agent_id = empty_workspace_agent_id


class Task:
    def __init__(self, task_id, chat_id, agents, owner, coordinator_agent, current_agent, memories,
                 parent_task_id=None, top_level_task_id=None, completed=False, created_at=None, depths=None,
                 is_top_level=False, top_level_task_max_depth=None, top_level_task_depth=None,
                 pending_children=0, inherited_memory_count=0, child_index=None,
                 pending_results=None):
        """
        Creates a new task in the database.
        :param task_id: The id of the task
        :param chat_id: The id of the original chat where the message was sent by the user to Stella
        :param agents: All the agents available in the Task.
        :param owner: The user's id
        :param coordinator_agent: The agent that the task was *originally* assigned to
        :param current_agent: The agent that the task is *currently* assigned to
        :param memories: The memories that the task has accumulated (note, parent tasks can also have memories)
        :param parent_task_id: The id of the parent task (if this is a subtask)
        :param top_level_task_id: The id of the top level task
        :param completed: Whether the task is completed or not
        :param created_at: The time the task was created in format "%Y-%m-%d %H:%M:%S" (YYYY-MM-DD HH:MM:SS)
        :param depths: Subtask depths
        :param is_top_level: Whether the task is a top level task or not
        :param top_level_task_max_depth: Max depth of the top level task
        :param top_level_task_depth: Current depth of the top level task
        :return: The created task
        """
        self.task_id = task_id
        self.chat_id = chat_id
        self.agents = agents
        self.owner = owner
        self.coordinator_agent = coordinator_agent
        self.current_agent = current_agent
        self.memories = memories
        self.parent_task_id = parent_task_id
        self.top_level_task_id = top_level_task_id
        self.completed = completed
        self.created_at = created_at
        self.depths = depths if depths is not None else {}
        self.is_top_level = is_top_level
        self.top_level_task_max_depth = top_level_task_max_depth if top_level_task_max_depth is not None else int(
            os.getenv('OVERALL_TASK_MAX_DEPTH', 9999))
        self.top_level_task_depth = top_level_task_depth if top_level_task_depth is not None else 0
        # Children spawned in the current turn that have not reported back yet.
        self.pending_children = pending_children or 0
        # How many of self.memories were inherited from the parent at creation time.
        # Only the entries after this point were produced by this task.
        self.inherited_memory_count = inherited_memory_count or 0
        # Position among the parent's children, so results can be collected in the
        # order they were delegated rather than the order they happen to finish.
        self.child_index = child_index
        # Results handed up by children, keyed by their child_index.
        self.pending_results = pending_results if pending_results is not None else {}

    @classmethod
    def create_top_level_task(cls, chat: Chat):
        """
        Creates a top level task
        (this is the top task in the tree that builds, and when it's done, the user gets a response.)
        :return:
        """
        # Load workspace
        try:
            workspace = db.get_workspace(chat.workspace_id)
        except Exception as e:
            raise Exception(f"Could not find workspace {chat.workspace_id}")

        # If the workspace has its own coordinator agent, use it. Otherwise fall back to
        # whichever default the host application configured with configure_default_agents():
        # the "empty workspace" one if the workspace has no agents connected yet, the
        # "general" one otherwise. Neither fallback is set by default -- stella_core has
        # no built-in notion of a coordinator or welcome agent, that is content the host
        # supplies (app/server.py configures its own, from stella_agents/StellaAgents/).
        coordinator_agent = workspace.coordinator_agent
        current_agent = workspace.coordinator_agent

        if not coordinator_agent:
            if len(workspace.agents) == 0:
                coordinator_agent = _default_empty_workspace_agent_id
                current_agent = _default_empty_workspace_agent_id
            else:
                coordinator_agent = _default_general_agent_id
                current_agent = _default_general_agent_id

        task_data = db.create_task(
            chat_id=chat.chat_id,
            agents=workspace.agents,
            owner=chat.owner,
            coordinator_agent=coordinator_agent,
            current_agent=current_agent,
            memories=[],
            is_top_level=True,
        )

        # Set the top level task id to the task id and update the task
        task_data['top_level_task_id'] = task_data['task_id']
        print(f"Created top level task {task_data['top_level_task_id']}")
        db.update_task_data(task_data)

        return cls(
            task_id=task_data['task_id'],
            chat_id=task_data['chat_id'],
            agents=task_data['agents'],
            owner=task_data['owner'],
            coordinator_agent=task_data['coordinator_agent'],
            current_agent=task_data['current_agent'],
            memories=task_data['memories'],
            parent_task_id=task_data.get('parent_task_id', None),
            top_level_task_id=task_data['top_level_task_id'],
            completed=task_data.get('completed', False),
            created_at=task_data.get('created_at', None),
            depths=task_data.get('depths', {}),
            is_top_level=task_data.get('is_top_level', False),
            top_level_task_max_depth=task_data.get('top_level_task_max_depth', None),
            top_level_task_depth=task_data.get('top_level_task_depth', None),
            pending_children=task_data.get('pending_children', 0),
            inherited_memory_count=task_data.get('inherited_memory_count', 0),
            child_index=task_data.get('child_index', None),
            pending_results=task_data.get('pending_results', {}),
        )

    @classmethod
    def load(cls, task_id):
        task_data = db.get_task_data(task_id)
        return cls(
            task_id=task_data['task_id'],
            chat_id=task_data['chat_id'],
            agents=task_data['agents'],
            owner=task_data['owner'],
            coordinator_agent=task_data['coordinator_agent'],
            current_agent=task_data['current_agent'],
            memories=task_data['memories'],
            parent_task_id=task_data.get('parent_task_id', None),
            top_level_task_id=task_data.get('top_level_task_id', None),
            completed=task_data.get('completed', False),
            created_at=task_data.get('created_at', None),
            depths=task_data.get('depths', {}),
            is_top_level=task_data.get('is_top_level', False),
            top_level_task_max_depth=task_data.get('top_level_task_max_depth', None),
            top_level_task_depth=task_data.get('top_level_task_depth', None),
            pending_children=task_data.get('pending_children', 0),
            inherited_memory_count=task_data.get('inherited_memory_count', 0),
            child_index=task_data.get('child_index', None),
            pending_results=task_data.get('pending_results', {}),
        )

    def check_max_depth_reached(self, selected_agent, top_level_task):
        result = {'is_max_depth_reached': False, 'message': ''}

        # Check overall max depth for top level task
        if self.top_level_task_id is not None:
            if top_level_task.top_level_task_max_depth is not None and top_level_task.top_level_task_depth is not None:
                if top_level_task.top_level_task_depth >= top_level_task.top_level_task_max_depth:
                    result['is_max_depth_reached'] = True
                    result['message'] = (f"Reached the overall call limit for this request "
                                         f"({top_level_task.top_level_task_depth}/"
                                         f"{top_level_task.top_level_task_max_depth}), stopping here.")
                    return result
            top_level_task.top_level_task_depth += 1

        # Check if the agent has a max depth and if it's reached
        max_depth_for_agent = selected_agent.max_depth
        if max_depth_for_agent is not None:
            agent_depth = top_level_task.depths.get(selected_agent.agent_id, 0)
            if agent_depth >= max_depth_for_agent:
                result['is_max_depth_reached'] = True
                result['message'] = (f"Reached the call limit for agent {selected_agent.name} "
                                     f"({agent_depth}/{max_depth_for_agent}), stopping here.")
            else:
                top_level_task.depths[selected_agent.agent_id] = agent_depth + 1

        db.update_task_data(top_level_task.to_dict())
        return result

    def _collect_child_results(self):
        """
        Folds the slots the children wrote into this task's memories, in the order the
        children were delegated in.

        Children finish in whatever order their work takes, so collecting on arrival would
        make the order of the results depend on timing -- and a slow agent's short answer
        landing behind a fast agent's large payload is exactly the arrangement the
        action-selection model tends to overlook. Runs when the task is picked up again,
        by which point every sibling has reported and nothing else is writing.
        """
        if not self.pending_results:
            return

        for key in sorted(self.pending_results, key=lambda k: int(k)):
            self.memories.extend(self.pending_results[key])

        print(f"[TASK] -- Collected results from {len(self.pending_results)} child slot(s) "
              f"in delegation order")
        self.pending_results = {}
        db.update_task_data(self.to_dict())

    def forward_memories_to_parent(self, current_agent):
        """
        Hands this task's results up into its slot on the parent, per the agent's flags.

        Two things are deliberate here. Only the entries this task produced are sent: the
        inherited ones are already in the parent, and returning them doubles the parent's
        memories on every round, which grew the prompt exponentially until it overflowed
        the context window. And they go into the slot matching the delegation order rather
        than straight onto the parent's memories, so results do not end up arranged by
        whichever sibling happened to finish first.
        :param current_agent: The agent that just finished, whose forwarding flags apply
        """
        if self.parent_task_id is None:
            return

        own_memories = self.memories[self.inherited_memory_count:]
        slot = self.child_index if self.child_index is not None else 0

        if current_agent.forward_all_memory_entries_to_parent:  # If all
            print(f"[TASK] -- Forwarding {len(own_memories)} own memory entrie(s) into "
                  f"slot {slot} of parent task {self.parent_task_id} "
                  f"(holding {len(self.memories)}, {self.inherited_memory_count} inherited)")
            db.store_child_result(self.parent_task_id, slot, own_memories)
        elif current_agent.forward_last_memory_to_parent:  # If last
            if own_memories:
                print(f"[TASK] -- Forwarding last memory into slot {slot} of parent task "
                      f"{self.parent_task_id}")
                print(f"[TASK] -- Last memory: {own_memories[-1]}")
                db.store_child_result(self.parent_task_id, slot, [own_memories[-1]])

    def _emit_progress(self, events, message):
        """
        Sends a progress line on the chat_information event.

        This is deliberately a different event from 'message': the client treats a chat
        message as the end of the request, and progress updates must not stop it waiting.
        :param events: The EventSink to report to
        :param message: Human readable text, e.g. "Asking WEATHER, BREWERY"
        """
        if not EMIT_PROGRESS_EVENTS:
            return
        try:
            events.send_progress(self.chat_id, message)
        except Exception as e:
            # Progress is cosmetic; never let it break the task it is reporting on.
            print(f"[TASK] !! Could not emit progress ({type(e).__name__}: {e})")

    def _report_to_parent(self, chat, events, give_up_message=None):
        """
        Hands control back up the tree once this task is finished.

        Every path that ends a task has to go through here. A child that returns without
        decrementing the counter leaves its parent waiting on a sibling that never
        arrives, which strands the chat on busy=True.
        :param chat: The chat this task belongs to
        :param events: Used to tell the user when a top level task gave up
        :param give_up_message: Set when the task stopped early rather than completing
        :return: The parent task id to re-queue, or None
        """
        if self.parent_task_id is None:
            # Nothing above us: close the request out so the user can send another message.
            if give_up_message:
                events.send_message(self.chat_id, give_up_message)
            print(f"[TASK] -- Top level task {self.task_id} finished without a response")
            chat.busy = False
            db.update_chat(chat)
            return None

        remaining = db.decrement_pending_children(self.parent_task_id)
        if remaining > 0:
            print(f"[TASK] -- Parent task {self.parent_task_id} still waiting on "
                  f"{remaining} sibling(s), not re-queuing it")
            return None

        print(f"[TASK] -- Last sibling done, re-queuing parent task {self.parent_task_id}")
        return self.parent_task_id

    def execute(self, agent_storage: AgentStorage, openai_client: OpenAIClient, events,
                request_builder: RequestBuilder):
        """
        Executes the task
        :param request_builder: Singleton instance of RequestBuilder (to build API Calls etc)
        :param agent_storage: Singleton instance of AgentStorage containing all agents (to load agents)
        :param openai_client: Singleton instance of OpenAIClient (to communicate with OpenAI)
        :param events: EventSink the runtime reports answers and progress to
        :return:
        """
        print(f"[Task] ---- Executing task: {self.task_id}")
        print(json.dumps(self.to_dict(), indent=4))

        # 1. Load the current agent
        # 2. Perform action selection
        #
        # 3.1 If action selection returns another Agent
        # 3.1.1 Load the agent
        # 3.1.2 Generate a new instruction for the agent
        # 3.1.3 Create a new task for the agent
        # 3.1.4 Return the new task
        #
        # 3.2 If action selection returns 0 -> Agent is done
        # 3.2.1 Generate a response
        # 3.2.2 If task is a top level task, send response to user (parent_task_id == None)
        # 3.2.2.1 Send message to user
        # 3.2.2.2 Save message in Chat object in database
        # 3.2.3 If task is a subtask, tell parent task
        # 3.2.3.1 Pass memories to parent task if necessary
        # 3.2.3.2 Re-queue the parent task
        chat = db.get_chat_by_id(self.chat_id)

        # 0. Fold in whatever the children reported while this task was parked.
        self._collect_child_results()

        # 1. Load the current agent
        current_agent = agent_storage.load(self.current_agent)
        print(f"[TASK] -- Loaded current agent: {current_agent.name}")

        print(f"Loading top level task {self.top_level_task_id}")
        top_level_task = Task.load(self.top_level_task_id)

        # Check if the agent's max_depth has been reached
        depth_check_result = self.check_max_depth_reached(current_agent, top_level_task)
        if depth_check_result['is_max_depth_reached']:
            print(f"[TASK] -- {depth_check_result['message']}")
            # A subtask that gives up still has to report back, otherwise its parent waits
            # on a sibling that will never arrive and the chat stays busy forever.
            return self._report_to_parent(chat, events, give_up_message=depth_check_result['message'])

        # 2. Perform action selection (if not disabled)
        selected_action = "0"  # Default to "Done"

        action_map = {}  # The action map is used by the agent to select an action, see Agent.select_action()
        i = 1  # Leave 0 for "done"

        # If this is a top level task, we add the coordinator agent's connections to the available agents
        if self.is_top_level:
            coordinator_agent = agent_storage.load(self.coordinator_agent)
            connections_available = coordinator_agent.connections_available
            self.agents.update(connections_available)

        print(f"[TASK] -- Building action map with agents: {self.agents}")
        for agent_id in self.agents.keys():
            print(f"[TASK] -- Loading agent {agent_id} for action selection")
            loaded_agent = agent_storage.load(agent_id)
            if loaded_agent is None:
                print(f"[TASK] -- Could not load agent {agent_id} for action selection, skipping")
                continue
            action_map[str(i)] = agent_storage.load(agent_id)
            i += 1
        print(f"[TASK] -- Action map: {action_map}")

        # Perform the action selection if it's not disabled, and the action map is not empty
        selected_actions = ["0"]
        if not current_agent.skip_action_selection and action_map:
            selected_actions = current_agent.select_actions(
                openai_client=openai_client,
                action_map=action_map,
                chat=chat if current_agent.wants_chat else None,
                memories=self.memories if current_agent.wants_memories else None,
            )

            # Agents are free to override selection, so never trust the result blindly.
            # An unknown key here would raise a KeyError below and take the worker down.
            valid = []
            for action in (str(a) for a in selected_actions):
                if action == "0" or action in action_map:
                    valid.append(action)
                else:
                    print(f"[TASK] -- {current_agent.name} returned invalid action "
                          f"'{action}', ignoring it")
            selected_actions = valid or ["0"]
        print(f"[TASK] -- {current_agent.name} Selected actions: {selected_actions}")

        # 3.1 If action selection returns other Agents
        delegated = [a for a in selected_actions if a != "0"]
        if delegated:
            # Record how many children have to report back before this task runs again.
            # It has to be stored before any child is queued, otherwise a child that
            # finishes immediately could decrement a counter that is still zero.
            self.pending_children = len(delegated)
            db.update_task_data(self.to_dict())

            names = ", ".join(action_map[a].name for a in delegated)
            self._emit_progress(events, f"Asking {names}…" if len(delegated) == 1
                                else f"Asking {names} in parallel…")

            child_task_ids = []
            for child_index, action in enumerate(delegated):
                # 3.1.1 Load the agent
                selected_agent = action_map[action]
                print(f"[TASK] -- Fetched agent information for: {selected_agent.name}")

                # 3.1.3 Create a new task for the new agent
                new_task_data = db.create_task(
                    chat_id=self.chat_id,
                    agents=selected_agent.connections_available,
                    owner=self.owner,
                    coordinator_agent=self.coordinator_agent,
                    current_agent=selected_agent.agent_id,
                    memories=self.memories,
                    parent_task_id=self.task_id,
                    top_level_task_id=self.top_level_task_id,
                    completed=False,
                    is_top_level=False,
                    # The child starts out holding the parent's memories as context. Record
                    # how many, so that on completion it forwards only what it produced
                    # itself instead of handing the parent its own entries back.
                    inherited_memory_count=len(self.memories),
                    # Slot this child writes its results into, so they come back in the
                    # order they were asked for rather than the order they complete.
                    child_index=child_index,
                )
                child_task_ids.append(new_task_data['task_id'])

            # 3.1.4 Return the new tasks. They run as siblings and the last one to finish
            # re-queues this task.
            print(f"[TASK] -- Spawned {len(child_task_ids)} child task(s): {child_task_ids}")
            return child_task_ids

        # 3.2 If action selection returns 0 -> Agent is done
        # 3.2.1 Generate a response (if not disabled)
        if not current_agent.skip_response:
            # Subtasks say nothing here: the parent already announced them when it
            # delegated, and repeating it just doubles every line.
            if self.parent_task_id is None:
                self._emit_progress(events, "Writing the answer…")
            response = current_agent.respond(
                openai_client=openai_client,
                request_builder=request_builder,
                chat=chat if current_agent.wants_chat else None,
                memories=self.memories if current_agent.wants_memories else None,
            )

            # Save the response in the task's memories
            self.memories.append(response)

            # 3.2.2 If task is a top level task, send response to user (parent_task_id == None)
            if self.parent_task_id is None:
                # 3.2.2.1 Send message to user
                print(f"[TASK] ---- Sending message to user: {response}")
                events.send_message(self.chat_id, response)

                # 3.2.2.2 Save message in Chat object in database
                chat.add_message(role="assistant", content=response)

                # Set chat busy state to false
                chat.busy = False
                db.update_chat(chat)
                return None
        self.completed = True
        db.update_task_data(self.to_dict())

        if self.parent_task_id is not None:
            self._emit_progress(events, f"{current_agent.name} done")

        # 3.2.3 If task is a subtask, tell parent task
        # 3.2.3.1 Pass memories to parent task if necessary
        self.forward_memories_to_parent(current_agent)

        if current_agent.on_completion is not None:
            print(f"[TASK] -- Agent {self.current_agent} has on_completion function, calling it")
            # Still passed as socketio=: agents were written against that keyword and call
            # .emit() on it, which EventSink keeps. Renaming it would break every existing
            # on_completion callback for no gain.
            current_agent.on_completion(socketio=events, chat_id=self.chat_id, chat=chat, memories=self.memories, openai_client=openai_client)

        # 3.2.3.2 Re-queue the parent task, but only once every sibling has reported back.
        return self._report_to_parent(chat, events)


    def __str__(self):
        return f"Task: {self.task_id}, " \
               f"Chat: {self.chat_id}, " \
               f"Agents: {self.agents}, " \
               f"Owner: {self.owner}, " \
               f"Coordinator Agent: {self.coordinator_agent}, " \
               f"Current Agent: {self.current_agent}, " \
               f"Parent Task: {self.parent_task_id}, " \
               f"Top Level Task: {self.top_level_task_id}, " \
               f"Memories: {self.memories}, " \
               f"Completed: {self.completed}, " \
               f"Created At: {self.created_at}, " \
               f"Depths: {self.depths}, " \
               f"Is Top Level: {self.is_top_level}, " \
               f"Top Level Task Max Depth: {self.top_level_task_max_depth}, " \
               f"Top Level Task Depth: {self.top_level_task_depth}."

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "chat_id": self.chat_id,
            "agents": self.agents,
            "owner": self.owner,
            "coordinator_agent": self.coordinator_agent,
            "current_agent": self.current_agent,
            "parent_task_id": self.parent_task_id,
            "top_level_task_id": self.top_level_task_id,
            "memories": self.memories,
            "completed": self.completed,
            "created_at": self.created_at,
            "depths": self.depths,
            "is_top_level": self.is_top_level,
            "top_level_task_max_depth": self.top_level_task_max_depth,
            "top_level_task_depth": self.top_level_task_depth,
            "pending_children": self.pending_children,
            "inherited_memory_count": self.inherited_memory_count,
            "child_index": self.child_index,
            "pending_results": self.pending_results,
        }
