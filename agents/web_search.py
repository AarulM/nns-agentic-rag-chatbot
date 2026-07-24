"""
Web search tool — live internet access for questions the Knowledge Base
can't answer (current events, general knowledge, external companies or
products, anything outside the HR / Safety / Operations docs).

Uses DuckDuckGo via `ddgs`, which needs no API key, so it works in local
testing with zero configuration. Results come back as title + snippet +
URL; the model reads those and answers, citing the source.
"""
import logging

from strands import tool

from trace_log import trace_queue

logger = logging.getLogger("nns.web_search")

_MAX_RESULTS = 5


@tool
def web_search(query: str) -> str:
    """Search the public web for current or general information that is not in
    the company knowledge base — current events, general facts, external
    companies, products, or public reference information.

    Do NOT use this for internal company HR, Safety, or Operations policy;
    those live in the knowledge base and belong to the specialist tools.

    Args:
        query: What to search the web for.

    Returns:
        The top results, each as a title, a snippet, and its source URL.
    """
    trace_queue.put(f"Searching the web → {query!r}")
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=_MAX_RESULTS))
    except Exception as error:
        # Never raise into the agent loop — a tool error makes the model
        # retry forever. Hand back a calm, final instruction instead.
        logger.warning("Web search failed for %r: %s", query, error)
        return (
            "The web search could not run right now. Tell the user you "
            "couldn't reach the web, and answer from your own knowledge if "
            "you can. Do not retry this search."
        )

    if not results:
        return f"No web results found for {query!r}."

    blocks = []
    for result in results:
        title = (result.get("title") or "").strip()
        snippet = (result.get("body") or "").strip()
        url = (result.get("href") or "").strip()
        blocks.append(f"{title}\n{snippet}\nSource: {url}")
    return "\n\n---\n\n".join(blocks)
