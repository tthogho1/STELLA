from abc import abstractmethod, ABC
from stella_core.models.workspace import Workspace
from stella_core.models.user import User
from stella_core.models.chat import Chat, ChatConnectionString, ChatMessageString


class DatabaseInterface(ABC):

    @abstractmethod
    def create_user(self, username, password) -> User:
        pass

    @abstractmethod
    def get_user_by_id(self, user_id) -> User:
        pass

    @abstractmethod
    def get_user_by_username(self, username) -> User:
        pass

    @abstractmethod
    def update_user(self, user: User) -> User:
        pass

    @abstractmethod
    def delete_user(self, user_id) -> None:
        pass

    @abstractmethod
    def get_user_workspaces(self, user_id) -> list[Workspace]:
        pass

    @abstractmethod
    def create_workspace(self, user_id, name, agents) -> Workspace:
        """
        Creates a new workspace in the database.
        :param user_id: The id of the user that owns the workspace
        :param name: The name of the workspace
        :param agents: The agents in the workspace
        :return: The created workspace
        """
        pass

    @abstractmethod
    def get_workspace(self, workspace_id) -> Workspace:
        pass

    @abstractmethod
    def delete_workspace(self, workspace_id) -> None:
        pass

    @abstractmethod
    def update_workspace(self, workspace: Workspace) -> Workspace:
        pass

    @abstractmethod
    def get_workspace_chats(self, workspace_id) -> list[str]:
        pass

    @abstractmethod
    def create_chat(self, workspace_id, user_id) -> Chat:
        pass

    @abstractmethod
    def get_chat_by_id(self, chat_id) -> Chat:
        pass

    @abstractmethod
    def update_chat(self, chat: Chat) -> Chat:
        pass

    @abstractmethod
    def delete_chat(self, chat_id) -> None:
        pass

    @abstractmethod
    def get_user_chats(self, user_id) -> list[Chat]:
        pass

    @abstractmethod
    def create_chat_connection_string(self, chat_id, string, created_at, expires_at, created_by) -> ChatConnectionString:
        pass

    @abstractmethod
    def get_chat_connection_string(self, string) -> ChatConnectionString:
        pass

    @abstractmethod
    def delete_chat_connection_string(self, string) -> None:
        pass

    @abstractmethod
    def create_chat_message_string(self, chat_id, string, created_at, expires_at, created_by) -> ChatMessageString:
        pass

    @abstractmethod
    def get_chat_message_string(self, string) -> ChatConnectionString:
        pass

    @abstractmethod
    def delete_chat_message_string(self, string) -> None:
        pass

    @abstractmethod
    def create_task(self, chat_id, agents, owner, coordinator_agent, current_agent, memories,
                    parent_task_id=None, top_level_task_id=None, completed=False, created_at=None, depths=None,
                    is_top_level=False, top_level_task_max_depth=None, top_level_task_depth=None,
                    pending_children=0, inherited_memory_count=0, child_index=None,
                    pending_results=None):
        pass

    @abstractmethod
    def get_task_data(self, task_id) -> dict:
        pass

    @abstractmethod
    def update_task_data(self, task_data) -> dict:
        pass

    @abstractmethod
    def delete_task(self, task_id) -> None:
        pass

    @abstractmethod
    def store_child_result(self, task_id, child_index, memories) -> None:
        """
        Records one child's results in its own slot on the parent, atomically.

        Siblings finish in whatever order their work happens to take, so appending
        straight onto the parent's memories makes the order of the results depend on
        timing: a slow agent's answer can end up behind a fast agent's large payload,
        where the action-selection model is liable to miss it. Each child writes to the
        slot matching the order it was delegated in, and the parent reads the slots back
        in that order. Writing has to be atomic because siblings write concurrently.
        :param task_id: The parent task
        :param child_index: Position of the child in the parent's delegation order
        :param memories: The memory strings this child produced
        """
        pass

    @abstractmethod
    def decrement_pending_children(self, task_id) -> int:
        """
        Decrements a task's outstanding-children counter and returns the new value.

        This is the join for parallel agent calls: every child decrements when it
        finishes, and only the one that takes the counter to zero re-queues the parent.
        Must be atomic, or two children finishing together both read 1 and the parent runs
        twice (or never).
        :param task_id: The parent task
        :return: The counter value after the decrement, never below zero
        """
        pass
