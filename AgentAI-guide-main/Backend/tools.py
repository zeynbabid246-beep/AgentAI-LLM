"""Agent tools: web search, Wikipedia lookup, and a safe calculator.

Each tool is a plain LangChain tool (docstring = description the LLM sees).
The calculator is deterministic: it parses arithmetic with Python's ``ast``
module and evaluates only whitelisted nodes — no LLM call, no ``eval``.
"""

import ast
import logging
import math
import operator
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small TTL cache for network tools: repeat questions skip the HTTP round-trip
# entirely (and don't consume any model quota — tool results come back free).
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 300.0
_CACHE_MAX_ENTRIES = 128
_tool_cache: Dict[str, Tuple[float, str]] = {}


def _cache_get(key: str) -> Optional[str]:
    entry = _tool_cache.get(key)
    if entry is None:
        return None
    stamp, value = entry
    if time.monotonic() - stamp > _CACHE_TTL_SECONDS:
        _tool_cache.pop(key, None)
        return None
    return value


def _cache_put(key: str, value: str) -> None:
    if len(_tool_cache) >= _CACHE_MAX_ENTRIES:
        oldest = min(_tool_cache, key=lambda k: _tool_cache[k][0])
        _tool_cache.pop(oldest, None)
    _tool_cache[key] = (time.monotonic(), value)

_WIKIPEDIA_SUMMARY_URL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKIPEDIA_EXTRACT_LIMIT = 1500
_HTTP_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# 1. Web search (DuckDuckGo via ddgs)
# ---------------------------------------------------------------------------

@tool
def duckduckgo_search(query: str) -> str:
    """Search the public web with DuckDuckGo.

    Use for current events, recent news, prices, weather, or any question
    that needs up-to-date information beyond your training data.

    Args:
        query: Short, focused search query (keywords work best).

    Returns:
        Top results as `title — snippet (url)` lines.
    """
    logger.info("Tool 'duckduckgo_search' called | query=%r", query)
    cache_key = f"web:{query.strip().lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("Tool 'duckduckgo_search' cache hit | query=%r", query)
        return cached
    try:
        try:
            from ddgs import DDGS  # current package name
            results = DDGS().text(query, max_results=5) or []
        except ImportError:
            from duckduckgo_search import DDGS  # legacy name (older mirrors)
            results = DDGS().text(query, max_results=5) or []
    except Exception:
        logger.exception("DuckDuckGo search failed")
        return "Web search failed. Tell the user the search service is temporarily unavailable."

    if not results:
        return f"No web results found for {query!r}. Try rephrasing or use your own knowledge."

    lines = [
        f"- {r.get('title', '').strip()} — {r.get('body', '').strip()} ({r.get('href', '').strip()})"
        for r in results
    ]
    rendered = "Web results:\n" + "\n".join(lines)
    _cache_put(cache_key, rendered)
    return rendered


# ---------------------------------------------------------------------------
# 2. Wikipedia lookup (REST summary API)
# ---------------------------------------------------------------------------

@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up an encyclopedic summary on Wikipedia.

    Use for factual, well-established knowledge about people, places,
    events, science, or history — not for current events.

    Args:
        topic: Page title or close approximation, e.g. "Albert Einstein".

    Returns:
        A plain-text summary of the matching Wikipedia article.
    """
    logger.info("Tool 'wikipedia_lookup' called | topic=%r", topic)
    cache_key = f"wiki:{topic.strip().lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        logger.info("Tool 'wikipedia_lookup' cache hit | topic=%r", topic)
        return cached
    url = _WIKIPEDIA_SUMMARY_URL.format(lang="en", title=requests.utils.quote(topic.strip().replace(" ", "_")))
    headers = {"User-Agent": "SmartAgentAPI/6.0 (educational demo)"}
    try:
        resp = requests.get(url, headers=headers, timeout=_HTTP_TIMEOUT)
    except requests.RequestException:
        logger.exception("Wikipedia request failed")
        return "Wikipedia lookup failed. Tell the user the lookup service is temporarily unavailable."

    if resp.status_code == 404:
        return f"No Wikipedia page found for {topic!r}. Try a different page title."

    if resp.status_code != 200:
        return f"Wikipedia returned HTTP {resp.status_code} for {topic!r}."

    data = resp.json()
    if data.get("type") == "disambiguation":
        return (
            f"{topic!r} is ambiguous on Wikipedia. Suggest the user disambiguate; "
            f"see {data.get('content_urls', {}).get('desktop', {}).get('page', url)}"
        )

    extract: str = data.get("extract", "") or "No summary available for this page."
    if len(extract) > _WIKIPEDIA_EXTRACT_LIMIT:
        extract = extract[:_WIKIPEDIA_EXTRACT_LIMIT].rsplit(" ", 1)[0] + "…"

    title = data.get("title", topic)
    rendered = f"Wikipedia — {title}:\n{extract}"
    _cache_put(cache_key, rendered)
    return rendered


# ---------------------------------------------------------------------------
# 3. Calculator (deterministic, AST-based — no LLM, no eval)
# ---------------------------------------------------------------------------

class CalculatorError(ValueError):
    """Raised when an expression is not a supported arithmetic expression."""


_BINARY_OPS: Dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: Dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_FUNCTIONS: Dict[str, Any] = {
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "ln": math.log,
    "log": math.log10,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
}

_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def safe_eval(expression: str) -> float:
    """Evaluate a pure-arithmetic expression and return a float.

    Supports ``+ - * / % ** //``, parentheses, unary +/-, constants
    (``pi``, ``e``, ``tau``) and the math functions in ``_FUNCTIONS``.
    Anything else (names, calls to unknown functions, attribute access,
    comparisons, strings...) raises ``CalculatorError``.
    """
    if not expression or not expression.strip():
        raise CalculatorError("Empty expression.")

    # Common user notation: 2^10 == 2**10
    normalized = expression.strip().replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid arithmetic expression: {expression!r}") from exc

    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise CalculatorError("Exponent too large.")
        try:
            return float(_BINARY_OPS[type(node.op)](left, right))
        except ZeroDivisionError as exc:
            raise CalculatorError("Division by zero.") from exc
        except OverflowError as exc:
            raise CalculatorError("Result too large.") from exc

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))

    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        args = [_eval_node(arg) for arg in node.args]
        if node.keywords:
            raise CalculatorError("Keyword arguments are not supported.")
        try:
            return float(_FUNCTIONS[node.func.id](*args))
        except (ValueError, OverflowError, TypeError) as exc:
            raise CalculatorError(f"Invalid arguments for {node.func.id}().") from exc

    raise CalculatorError(f"Unsupported element in expression: {ast.dump(node)[:80]}")


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression exactly and instantly.

    Use for any arithmetic: percentages, unit conversions, large numbers,
    roots, powers, trigonometry. Do NOT do arithmetic in your head — always
    use this tool so results are exact.

    Args:
        expression: Pure arithmetic expression, e.g. "(1250*12)/3" or
            "sqrt(144) + 2**10". Supported: + - * / % ** // ^ ( ),
            constants pi/e/tau, functions sqrt, ln, log, log2, exp, abs,
            round, floor, ceil, sin, cos, tan.

    Returns:
        The computed result.
    """
    logger.info("Tool 'calculator' called | expression=%r", expression)
    try:
        result = safe_eval(expression)
    except CalculatorError as exc:
        return f"Calculator error: {exc}"
    except RecursionError:
        return "Calculator error: expression is too deeply nested."

    if math.isinf(result):
        return "Calculator error: result is infinite."
    if result == int(result) and abs(result) < 1e15:
        return f"{expression.strip()} = {int(result)}"
    return f"{expression.strip()} = {result}"


def get_tools() -> List:
    """Return the tool set exposed to the agent."""
    return [duckduckgo_search, wikipedia_lookup, calculator]
