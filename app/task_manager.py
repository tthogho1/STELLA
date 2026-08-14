import json
import queue
import threading
import traceback

from app.agent_storage import AgentStorage
from app.db import db
from app.models.task import Task
from app.utils.request_builder import RequestBuilder
from app.openai_client import OpenAIClient


class Worker(threading.Thread):
    def __init__(self, task_queue, manager, agent_storage, openai_client, request_builder):
        threading.Thread.__init__(self)
        self.task_queue = task_queue
        self.manager = manager
        self.agent_storage = agent_storage
        self.openai_client = openai_client
        self.request_builder = request_builder
        self.daemon = True

    def run(self):
        while True:
            task_id = self.task_queue.get()

            # Check if chat has been picked up by another worker
            if task_id in self.manager.active_tasks:
                self.task_queue.task_done()
                continue

            print(f"[TaskQueue] -- Worker {self.name} picked up task {task_id}")
            self.manager.active_tasks.append(task_id)

            try:
                task = Task.load(task_id)
                new_task_id = task.execute(self.agent_storage, self.openai_client, self.manager.socketio, self.request_builder)
                if new_task_id is not None:
                    self.manager.add_task(new_task_id)
            except Exception as e:
                # Never let an exception escape: it would kill this thread for good,
                # shrinking the worker pool and leaving the chat stuck on busy=True.
                print(f"[TaskQueue] !! Worker {self.name} failed task {task_id}: {e}")
                traceback.print_exc()
                self._release_chat(task_id)
            finally:
                self.task_queue.task_done()
                print(f"[TaskQueue] ---- Worker {self.name} finished task {task_id}")
                if task_id in self.manager.active_tasks:
                    self.manager.active_tasks.remove(task_id)

    def _release_chat(self, task_id):
        """
        Clears the busy flag on the chat a failed task belonged to, so the user is able to
        send another message, and lets them know the request could not be completed.
        :param task_id: The id of the task that failed
        """
        try:
            chat = db.get_chat_by_id(db.get_task_data(task_id)['chat_id'])
            chat.busy = False
            db.update_chat(chat)
            self.manager.socketio.emit(
                'message',
                "Sorry, something went wrong while processing your request. Please try again.",
                room=chat.chat_id,
                namespace='/chat'
            )
        except Exception as e:
            print(f"[TaskQueue] !! Could not release the chat for task {task_id}: {e}")


class TaskManager:
    def __init__(self, num_workers, socketio, agent_storage):
        self.socketio = socketio
        self.task_queue = queue.Queue()
        self.agent_storage = agent_storage
        self.openai_client = OpenAIClient()
        self.request_builder = RequestBuilder(openai_client=self.openai_client)
        self.workers = [Worker(self.task_queue, self, agent_storage=self.agent_storage, openai_client=self.openai_client, request_builder=self.request_builder) for _ in range(num_workers)]
        self.active_tasks = []
        for worker in self.workers:
            print(f"[TaskQueue] Starting worker {worker}")
            worker.start()

    def add_task(self, task_id):
        print(f"[TaskQueue] Adding task to queue: {task_id}")
        self.task_queue.put(task_id)
