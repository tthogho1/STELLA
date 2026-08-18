
import json
import queue
import threading
import traceback

from stella_core.db import db
from stella_core.models.chat import Chat
from stella_core.task_manager import TaskManager, Task


class Worker(threading.Thread):
    def __init__(self, chat_queue, manager, task_manager, events):
        threading.Thread.__init__(self)
        self.chat_queue = chat_queue
        self.manager = manager
        self.task_manager = task_manager
        self.events = events
        self.daemon = True

    def run(self):
        """
        Keep in mind that this function only gets triggered when a user sends a message.
        It is never triggered by the system.
        :return:
        """
        while True:
            chat_id = self.chat_queue.get()

            # Check if chat has been picked up by another worker
            if chat_id in self.manager.active_chats:
                self.chat_queue.task_done()
                continue

            print(f"[ChatQueue] -- Worker {self.name} picked up chat {chat_id}")
            self.manager.active_chats.append(chat_id)

            try:
                chat = db.get_chat_by_id(chat_id)

                if chat.busy:
                    # The payload used to be commented out, so the client received an
                    # event with no data and printed nothing useful.
                    self.events.send_progress(
                        chat_id,
                        "This chat is still processing your previous request, please wait.",
                        kind="busy")

                chat.busy = True
                db.update_chat(chat)

                task = Task.create_top_level_task(chat)
                self.task_manager.add_task(task.task_id)
            except Exception as e:
                # Same reasoning as in TaskManager's worker: an escaping exception would
                # kill this thread and leave the chat permanently marked as busy.
                print(f"[ChatQueue] !! Worker {self.name} failed chat {chat_id}: {e}")
                traceback.print_exc()
                self._release_chat(chat_id)
            finally:
                self.chat_queue.task_done()
                print(f"[ChatQueue] ---- Worker {self.name} finished chat {chat_id}")
                if chat_id in self.manager.active_chats:
                    self.manager.active_chats.remove(chat_id)

    def _release_chat(self, chat_id):
        """
        Clears the busy flag so the user can send another message, and tells them that the
        request could not be started.
        :param chat_id: The id of the chat that failed
        """
        try:
            chat = db.get_chat_by_id(chat_id)
            chat.busy = False
            db.update_chat(chat)
            self.events.send_message(
                chat_id,
                "Sorry, something went wrong while starting your request. Please try again.")
        except Exception as e:
            print(f"[ChatQueue] !! Could not release the chat {chat_id}: {e}")


class ChatQueue:
    def __init__(self, num_workers, events, agent_storage):
        self.num_workers = num_workers
        self.events = events
        self.agent_storage = agent_storage
        self.chat_queue = queue.Queue()
        self.task_manager = TaskManager(num_workers=5, events=events, agent_storage=agent_storage)
        self.workers = [Worker(self.chat_queue, self, task_manager=self.task_manager, events=events) for _ in range(num_workers)]
        self.active_chats = []
        for worker in self.workers:
            print(f"[ChatQueue] Starting worker {worker}")
            worker.start()

    def add_chat(self, chat_id):
        # Check if chat is already in queue
        if chat_id in self.active_chats:
            print(f"[ChatQueue] Chat already in queue: {chat_id}")
            return
        print(f"[ChatQueue] Chat added to queue: {chat_id}")
        self.chat_queue.put(chat_id)
