"""
Exposes the LangChain `search_cities` tool as a STELLA agent.

STELLA has no notion of a LangChain registry: AgentStorage only instantiates Agent
subclasses found under the directories it is given, so a tool is reached by wrapping it.
The tool itself is untouched in stella_agents/tools/city_search.py and can still be used
from LangChain directly.
"""
import json

from stella_core.models.agent import Agent
from stella_core.models.chat import Chat
from stella_core.openai_client import OpenAIClient
from stella_core.utils.request_builder import RequestBuilder
from stella_agents.tools.city_search import search_cities


class CitySearchAgent(Agent):
    """
    Travel destination search over the Wikivoyage index.
    """

    # A whole page of passages would be re-sent on every later agent call in the tree,
    # which is how a large payload ends up crowding a short result out of the coordinator's
    # attention. Keep what is handed back short and readable.
    max_results_in_memory = 3
    max_summary_chars = 1200

    def __init__(self):
        super().__init__(
            agent_id='city_search_agent',
            name='CITY_SEARCH',
            short_description='Find travel destinations and city recommendations',
            # A leaf: it has nothing to delegate to, so skip the selection LLM call.
            skip_action_selection=True,
            # Hand the result up to the coordinator.
            forward_all_memory_entries_to_parent=True,
        )

    def _extract_query(self, openai_client: OpenAIClient, chat: Chat = None, memories=None):
        """
        Turns the conversation into a search query.

        The tool takes free text, and the user's message is usually close enough already,
        but it can be conversational ("we were thinking somewhere warm, maybe by the
        sea?"). Same approach the weather demo uses to pull out a city.
        """
        if chat is None and not memories:
            return ""

        messages = [
            {
                "role": "system",
                "content": "Extract a travel destination search query from the "
                           "conversation. Reply with the query only, no other text, "
                           'e.g. "coastal cities in Japan" or "Osaka".'
            },
            {
                "role": "user",
                "content": f"{self._construct_chat_string(chat) if chat else ''}"
                           f"{self._construct_memory_string(memories) if memories else ''}"
            }
        ]

        query = openai_client.chat_completion(
            messages=messages,
            model=self.model_for_action_selection,
        )
        return (query or "").strip().strip('"')

    def _format(self, raw):
        """
        Turns the tool's JSON envelope into the one string that goes into memories.

        The tool reports its own failures in an "error" field rather than raising, so
        those have to be surfaced here or the coordinator sees an empty result and simply
        calls the agent again.
        """
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return str(raw)[:self.max_summary_chars]

        if data.get("error"):
            return (f"The city search could not be run: {data['error']}. "
                    f"Tell the user this lookup is unavailable and do not retry it.")

        if not data.get("results"):
            return (f"The city search found nothing for '{data.get('query', '')}'. "
                    f"Tell the user and do not retry the same search.")

        lines = [f"City search results for '{data.get('query', '')}':"]
        summary = (data.get("summary") or "").strip()
        if summary:
            lines.append(f"Summary: {summary[:self.max_summary_chars]}")

        for result in data["results"][:self.max_results_in_memory]:
            title = result.get("title", "N/A")
            url = result.get("url")
            lines.append(f"- {title}" + (f" ({url})" if url else ""))

        return "\n".join(lines)

    def respond(self, openai_client: OpenAIClient, request_builder: RequestBuilder,
                chat: Chat = None, memories=None):
        query = self._extract_query(openai_client, chat, memories)
        if not query:
            return ("The city search had no query to run. Tell the user what "
                    "destination or kind of place they are looking for.")

        print(f"[AGENT] {self.name} searching for: {query!r}")

        try:
            # .invoke() is the LangChain calling convention; the tool validates its own
            # arguments and returns a JSON string.
            raw = search_cities.invoke({"query": query})
        except Exception as e:
            # The tool catches its own failures, so reaching here means something in
            # LangChain itself went wrong. Never let it take the worker down.
            print(f"[AGENT] {self.name} tool call failed ({type(e).__name__}: {e})")
            return (f"The city search failed to run ({type(e).__name__}). Tell the user "
                    f"this lookup is unavailable and do not retry it.")

        return self._format(raw)
