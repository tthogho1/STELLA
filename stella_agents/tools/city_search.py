"""City search tool backed by Databricks Vector Search + an LLM summary.

Runs a hybrid similarity search over the Wikivoyage index, then asks a
Databricks serving endpoint to summarize the retrieved passages. Mirrors the
retrieval/summarize flow used in the databrikcstest sample.
"""

import json
import os

from langchain_core.tools import tool

# ── Config (overridable via environment variables) ───────────────────────────
LLM_MODEL = os.environ.get("CITIES_LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")
ENDPOINT_NAME = os.environ.get("CITIES_ENDPOINT_NAME", "wikivoyage_seach_endpoint")
INDEX_NAME = os.environ.get("CITIES_INDEX_NAME", "workspace.default.wikivoyage_index")
# Comma-separated list, e.g. "content,title".
COLUMNS = [
    c.strip()
    for c in os.environ.get("CITIES_COLUMNS", "content,title,url").split(",")
    if c.strip()
]
NUM_RESULTS = int(os.environ.get("CITIES_NUM_RESULTS", "3"))


@tool
def search_cities(query: str) -> str:
    """Find travel destinations for a query. USE THIS whenever the user asks
    where to go, for place/city ideas, or for destination recommendations.

    Runs a hybrid similarity search over the Wikivoyage travel index and
    returns a short LLM-written summary alongside the top matching passages.

    Args:
        query: Free-text search, e.g. "coastal cities in Japan" or "Osaka".

    Returns:
        A JSON string with the query, the top results, and a summary.
    """
    if not query or not query.strip():
        return json.dumps({"query": query, "error": "empty query", "results": []})

    workspace_url = os.environ.get("WORKSPACE_URL")
    access_token = os.environ.get("DATABRICKS_TOKEN")
    missing = [
        name
        for name, value in (
            ("WORKSPACE_URL", workspace_url),
            ("DATABRICKS_TOKEN", access_token),
        )
        if not value
    ]
    if missing:
        return json.dumps(
            {
                "query": query,
                "error": f"missing env var(s): {', '.join(missing)}",
                "results": [],
            }
        )

    # Lazily import the optional Databricks/OpenAI deps so the agents package
    # stays importable when they (or their credentials) are unavailable.
    try:
        from databricks.ai_search.client import VectorSearchClient
        from openai import OpenAI
    except ImportError as exc:
        return json.dumps({"query": query, "error": f"missing dependency: {exc}"})

    # ── Retrieve: hybrid similarity search over the index ─────────────────────
    try:
        vsc = VectorSearchClient(
            workspace_url=workspace_url,
            personal_access_token=access_token,
            disable_notice=True,
        )
        index = vsc.get_index(
            endpoint_name=ENDPOINT_NAME,
            index_name=INDEX_NAME,
        )
        search = index.similarity_search(
            num_results=NUM_RESULTS,
            columns=COLUMNS,
            query_text=query,
            query_type="HYBRID",
        )
    except Exception as exc:  # noqa: BLE001 — surface any client/network error
        return json.dumps({"query": query, "error": f"city search failed: {exc}"})

    hits = search.get("result", {}).get("data_array", [])
    col_names = [
        col.get("name") for col in search.get("manifest", {}).get("columns", [])
    ]
    if not hits:
        return json.dumps({"query": query, "count": 0, "results": [], "summary": ""})

    results = []
    context_parts = []
    for hit in hits:
        row = dict(zip(col_names, hit))
        title = row.get("title", "N/A")
        content = str(row.get("content", ""))
        results.append(
            {"title": title, "url": row.get("url"), "content": content[:300]}
        )
        context_parts.append(f"Title: {title}\n{content}")

    # ── Summarize the retrieved passages with the Databricks LLM ──────────────
    context_text = "\n\n---\n\n".join(context_parts)
    prompt = (
        f'Based on the following travel information retrieved for the query "{query}",\n'
        "write a concise and engaging summary (3-5 sentences) highlighting the key "
        "points.\n\n"
        f"Retrieved information:\n{context_text}\n\nSummary:"
    )

    try:
        llm_client = OpenAI(
            api_key=access_token,
            base_url=f"{workspace_url}/serving-endpoints",
        )
        chat_response = llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        summary = chat_response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 — degrade to results without a summary
        summary = f"(summary unavailable: {exc})"

    return json.dumps(
        {
            "query": query,
            "count": len(results),
            "results": results,
            "summary": summary,
        }
    )
