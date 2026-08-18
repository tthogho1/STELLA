from openai import OpenAI, DefaultHttpxClient
import os
import ssl

import threading
from queue import Queue


def _int_env(name, default):
    try:
        return max(int(os.getenv(name, default)), 0)
    except (TypeError, ValueError):
        print(f"[OPENAI] {name} is not a number, falling back to {default}")
        return int(default)


# Retries cover rate limits (429, honouring Retry-After), 5xx and connection errors, with
# exponential backoff and jitter, and are applied per request inside the SDK. Errors that
# retrying cannot fix -- a bad key, or a prompt over the context window -- still fail
# immediately. Without a timeout a stalled request holds one of the workers below forever.
OPENAI_MAX_RETRIES = _int_env("OPENAI_MAX_RETRIES", 4)
OPENAI_TIMEOUT_SECONDS = _int_env("OPENAI_TIMEOUT_SECONDS", 120)
# How many requests may be in flight at once. Each worker holds one, so this is also the
# lever for staying inside an account's rate limits.
OPENAI_MAX_WORKERS = _int_env("OPENAI_MAX_WORKERS", 30) or 1


def _build_client():
    """
    Builds the OpenAI client.

    The server runs under gevent.monkey.patch_all(), and by default the SDK's HTTP layer
    verifies TLS through `truststore`, whose SSLContext subclass recurses infinitely
    against gevent's patched `ssl` module (RecursionError -> APIConnectionError on every
    request). Handing the client an ordinary certifi-backed context keeps truststore out
    of the path. Without monkey patching this is simply the normal verification path.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    settings = {"max_retries": OPENAI_MAX_RETRIES, "timeout": OPENAI_TIMEOUT_SECONDS}
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
        return OpenAI(api_key=api_key, http_client=DefaultHttpxClient(verify=context), **settings)
    except Exception as e:
        print(f"[OPENAI] Could not build a certifi-backed HTTP client ({type(e).__name__}: {e}), "
              f"falling back to SDK defaults")
        return OpenAI(api_key=api_key, **settings)


client = _build_client()

# Default models. Overridable per agent, and through app/.env so that they can be updated
# when OpenAI retires a model without having to touch the code.
DEFAULT_ACTION_SELECTION_MODEL = os.getenv("OPENAI_MODEL_ACTION_SELECTION", "gpt-4o-mini")
DEFAULT_RESPONSE_MODEL = os.getenv("OPENAI_MODEL_RESPONSE", "gpt-4o")


class OpenAIQuery:
    def __init__(self, messages, model, query_type="chat_completion", tools=None, tool_choice=None,
                 response_format=None):
        self.done = threading.Event()
        self.exception = None
        self.result = None
        self.messages = messages
        self.model = model
        self.query_type = query_type
        self.tools = tools
        self.tool_choice = tool_choice
        self.response_format = response_format


class OpenAIClient:
    """
    Responsible for calling OpenAI API and returning the results
    This is done using a queue,
    and in a way that respects the rate limits of the API.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:  # Singleton
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, max_workers: int = None):
        if hasattr(self, '_initialized'):  # Singleton
            return
        self._initialized = True

        max_workers = OPENAI_MAX_WORKERS if max_workers is None else max_workers

        self.max_workers = max_workers

        self.current_queries = set()
        self.query_queue = Queue()

        self._init_worker_threads()

    def _init_worker_threads(self):
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._worker)
            worker.daemon = True
            worker.start()

    def _worker(self):
        while True:
            query = self.query_queue.get()
            try:
                if not query:
                    continue
                if query in self.current_queries:
                    continue
                self.current_queries.add(query)
                self._query(query)
            except Exception as e:
                query.exception = e
                query.done.set()

    def _query(self, query: OpenAIQuery):
        """
        Queries OpenAI API
        :param query:
        :return:
        """
        # Retries and backoff are handled by the SDK client (OPENAI_MAX_RETRIES), and the
        # size of the worker pool caps how many requests are in flight at once.
        try:
            if query.query_type == "chat_completion":

                print(f"[OPENAI Query]\nModel: {query.model}\n{query.messages}")

                # response_format is only sent when asked for, so models that predate it
                # keep working unchanged.
                extra = {}
                if query.response_format is not None:
                    extra["response_format"] = query.response_format

                response = client.chat.completions.create(model=query.model,
                                                          messages=query.messages,
                                                          **extra)
                query.result = response
                query.done.set()
            elif query.query_type == "tool_call":

                print(f"[OPENAI Tool Query]\nModel: {query.model}\n{query.messages}")

                response = client.chat.completions.create(model=query.model,
                                                          messages=query.messages,
                                                          tools=query.tools,
                                                          tool_choice=query.tool_choice or "auto")
                query.result = response
                query.done.set()
            else:
                raise NotImplementedError(f"Query type {query.query_type} not implemented")
        except Exception as e:
            query.exception = e
            query.done.set()

    def _run(self, query: OpenAIQuery):
        """
        Puts a query on the queue, blocks until a worker has resolved it and returns the
        raw API response. Raises whatever the worker caught.
        :param query: The query to run
        :return: The raw OpenAI response object
        """
        # Add query to queue
        self.query_queue.put(query)

        # Wait for query to finish
        query.done.wait()

        # Remove query from current queries
        self.current_queries.discard(query)

        # Check if there was an exception
        if query.exception:
            raise query.exception

        return query.result

    def chat_completion(self, messages, model, response_format=None):
        """
        :param messages: Messages to send
        :param model: Model to use
        :param response_format: Optional OpenAI response_format, e.g. {"type": "json_object"}
                                or a json_schema block, to constrain the reply
        :return: The message content
        """
        response = self._run(OpenAIQuery(messages, model, query_type="chat_completion",
                                         response_format=response_format))

        # Return result
        return response.choices[0].message.content

    def tool_call(self, messages, model, tools, tool_choice="required"):
        """
        Same as chat_completion, but exposes a set of tools to the model and returns the
        message it produced so the caller can read message.tool_calls.

        Unlike asking the model to answer with a bare index, the API constrains the reply
        to one of the supplied tool names, so there is nothing to parse out of free text.
        :param messages: Messages to send
        :param model: Model to use
        :param tools: Tool definitions in OpenAI's function-tool format
        :param tool_choice: "required" forces the model to pick one of the tools
        :return: The assistant message (has .tool_calls and .content)
        """
        response = self._run(OpenAIQuery(messages, model, query_type="tool_call",
                                         tools=tools, tool_choice=tool_choice))

        return response.choices[0].message
