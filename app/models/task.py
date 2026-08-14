import json
import os
from dotenv import load_dotenv

from app.db import db
from app.models.chat import Chat
from app.agent_storage import AgentStorage
from app.openai_client import OpenAIClient
from app.utils.request_builder import RequestBuilder

# Load the environment variables from the .env file
load_dotenv()


class Task:
    def __init__(self, task_id, chat_id, agents, owner, coordinator_agent, current_agent, memories,
                 parent_task_id=None, top_level_task_id=None, completed=False, created_at=None, depths=None,
                 is_top_level=False, top_level_task_max_depth=None, top_level_task_depth=None,
                 pending_children=0):
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

        # Check if there is a coordinator agent assigned to the workspace. If so assign it. Otherwise:
        # If there are no available agents, set the coordinator_agent, current_agent to "stella_welcome_agent",
        # this is a special Agent which is configured for helping the user get started if they have yet to connect
        # any agents to their workspace. /agents/StellaWelcomeAgent.py
        # If there are agents in the workspace (any), then we select the "stella_coordinator_agent"
        # which is the default response agent. /agents/StellaCoordinatorAgent.py
        coordinator_agent = workspace.coordinator_agent
        current_agent = workspace.coordinator_agent

        if not coordinator_agent:
            if len(workspace.agents) == 0:
                coordinator_agent = "stella_welcome_agent"
                current_agent = "stella_welcome_agent"
            else:
                coordinator_agent = "stella_coordinator_agent"
                current_agent = "stella_coordinator_agent"

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

    def _report_to_parent(self, chat, socketio, give_up_message=None):
        """
        Hands control back up the tree once this task is finished.

        Every path that ends a task has to go through here. A child that returns without
        decrementing the counter leaves its parent waiting on a sibling that never
        arrives, which strands the chat on busy=True.
        :param chat: The chat this task belongs to
        :param socketio: Used to tell the user when a top level task gave up
        :param give_up_message: Set when the task stopped early rather than completing
        :return: The parent task id to re-queue, or None
        """
        if self.parent_task_id is None:
            # Nothing above us: close the request out so the user can send another message.
            if give_up_message:
                socketio.emit('message', give_up_message, room=self.chat_id, namespace='/chat')
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

    def execute(self, agent_storage: AgentStorage, openai_client: OpenAIClient, socketio,
                request_builder: RequestBuilder):
        """
        Executes the task
        :param request_builder: Singleton instance of RequestBuilder (to build API Calls etc)
        :param agent_storage: Singleton instance of AgentStorage containing all agents (to load agents)
        :param openai_client: Singleton instance of OpenAIClient (to communicate with OpenAI)
        :param socketio: Singleton instance of SocketIO (to communicate with the user)
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
            return self._report_to_parent(chat, socketio, give_up_message=depth_check_result['message'])

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

            child_task_ids = []
            for action in delegated:
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
                )
                child_task_ids.append(new_task_data['task_id'])

            # 3.1.4 Return the new tasks. They run as siblings and the last one to finish
            # re-queues this task.
            print(f"[TASK] -- Spawned {len(child_task_ids)} child task(s): {child_task_ids}")
            return child_task_ids

        # 3.2 If action selection returns 0 -> Agent is done
        # 3.2.1 Generate a response (if not disabled)
        if not current_agent.skip_response:
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
                socketio.emit('message', response, room=self.chat_id, namespace='/chat')

                # 3.2.2.2 Save message in Chat object in database
                chat.add_message(role="assistant", content=response)

                # Set chat busy state to false
                chat.busy = False
                db.update_chat(chat)
                return None
        self.completed = True
        db.update_task_data(self.to_dict())

        # 3.2.3 If task is a subtask, tell parent task
        # 3.2.3.1 Pass memories to parent task if necessary.
        # Siblings running in parallel forward to the same parent, so the append has to
        # happen inside the database rather than as load -> extend -> save here.
        if self.parent_task_id is not None:
            if current_agent.forward_all_memory_entries_to_parent:  # If all
                print(f"[TASK] -- Forwarding all memories to parent task {self.parent_task_id}")
                print(f"[TASK] -- Memories: {self.memories}")
                db.append_task_memories(self.parent_task_id, self.memories)
            elif current_agent.forward_last_memory_to_parent:  # If last
                if self.memories:
                    print(f"[TASK] -- Forwarding last memory to parent task {self.parent_task_id}")
                    print(f"[TASK] -- Last memory: {self.memories[-1]}")
                    db.append_task_memories(self.parent_task_id, [self.memories[-1]])

        if current_agent.on_completion is not None:
            print(f"[TASK] -- Agent {self.current_agent} has on_completion function, calling it")
            current_agent.on_completion(socketio=socketio, chat_id=self.chat_id, chat=chat, memories=self.memories, openai_client=openai_client)

        # 3.2.3.2 Re-queue the parent task, but only once every sibling has reported back.
        return self._report_to_parent(chat, socketio)


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
        }