# Smart Agent API — Study / Interview Questions & Answers

Question bank covering everything the project does: architecture, multi-provider LLMs,
the agent tool loop, SSE streaming, hardening, and testing. Short answers — each is
2–5 sentences of what a strong answer contains.

---

## A. Architecture & Multi-Provider Design

**1. Why do five providers (Gemini, Groq, Ollama, OpenRouter, Cerebras) work through one code path?**
They all expose an OpenAI-compatible chat-completions API, so a single `ChatOpenAI`
instance — parameterized by base URL, API key, and model name — serves all of them.
Only three things vary per provider, so they live in a config registry (`PROVIDERS`):
the key env var, the base URL, and a default model. No provider-specific code exists
in the agent loop itself.

**2. How does a user switch providers? What must restart?**
Set `LLM_PROVIDER` (and the matching `*_API_KEY`) in `Backend/.env` and restart
uvicorn, because settings are lru-cached at startup and the agent is a lazy singleton.
Nothing else changes: tools, SSE protocol, memory, and rate limiting are
provider-agnostic.

**3. Where do provider-specific defaults live, and what does each entry contain?**
In the `PROVIDERS` dict in `config.py`. Each entry has `env_key` (which Settings
attribute holds the key, `"none"` for Ollama), `base_url`, `default_model`, and a
human `label`. `Settings` exposes derived properties (`provider_id`,
`provider_base_url`, `provider_api_key`, `resolved_model`, `missing_key_env`) that
resolve these per selected provider.

**4. Why is Ollama "always configured" and what placeholder does the code send as its key?**
It runs locally with no authentication, so there is no key to check. The OpenAI SDK
rejects an empty api_key string, so `provider_api_key` returns the placeholder
`"ollama"`, which the local server ignores.

**5. What happens if `LLM_PROVIDER` is misspelled?**
`provider_id` lowercases the value and falls back to `"gemini"` if it isn't in the
registry — the app degrades to a known provider instead of crashing. `MODEL_NAME`
still overrides whatever default the effective provider has.

**6. Why must `reasoning_effort` be sent only to Gemini?**
It's a Gemini-3-specific kwarg ("thinking" budget). Other providers reject unknown
request-body fields with HTTP 422, so `_build_openai_compat_agent` adds it to
`model_kwargs` only when `provider_id == "gemini"`.

**7. Why did the project keep a per-provider default model instead of one global model name?**
Model names are provider-specific (`qwen/qwen3.8-27b` means nothing to Cerebras), and
catalogs drift — Groq retired `llama-3.3-70b-versatile` in Sept 2026. A per-provider
default keeps `MODEL_NAME` optional while an explicit `MODEL_NAME` still overrides it.

**8. How should the app react when a configured model 404s ("model_not_found")?**
`_friendly_error` maps 404 + model-not-found phrasing to an actionable message: the
configured model isn't offered by that provider; set `MODEL_NAME` to a model the
provider offers (their `/v1/models` list shows it) and restart.

---

## B. Agent Design & the Text-Protocol Tool Loop

**9. What are the two execution tiers, and when does each run?**
Tier 1 "langgraph": LangGraph prebuilt ReAct agent + `ChatGoogleGenerativeAI` with a
checkpointer — only for the Gemini provider when langgraph/langchain-google-genai are
installed. Tier 2 "openai_compat": a hand-rolled text-protocol tool loop over
`ChatOpenAI` — used for every non-Gemini provider, and for Gemini when tier-1
packages are missing. Both emit identical events.

**10. Why does the fallback loop use a text protocol (`ACTION:/INPUT:`/`FINAL:`) instead of OpenAI native tool calls?**
When it was built, Gemini 3 required `thought_signature` echoes on tool-call history
that the era's OpenAI-compat adapters dropped, causing 400s. A plain-text protocol
needs no provider tool-call support at all, which also made multi-provider support
nearly free. Trade-off: the model must follow formatting instructions reliably.

**11. Walk through one tool round in the fallback loop.**
Model reply is parsed by `_parse_action` (regex for `ACTION: <tool>` + `INPUT: <arg>`).
If found, the loop emits `tool_start`, runs the tool via
`run_in_executor` (blocking HTTP off the event loop), emits `tool_end`, appends the
reply and `OBSERVATION: <result truncated to 2000 chars>` to the conversation, and
loops (max `AGENT_MAX_ITERATIONS` steps). Without `ACTION:`, the reply is final.

**12. How is session memory implemented in tier 2?**
A per-session bounded `deque(maxlen=16)` in `_histories`, holding alternating
human/ai texts; replayed as `HumanMessage`/`AIMessage` before each new user message.
Tier 1 instead relies on LangGraph's `InMemorySaver` checkpointer keyed by
`thread_id`.

**13. What protections exist against prompt-injected "tool" instructions?**
Tool names are matched against the registered set — unknown ones return an
"Unknown tool" observation instead of executing. The calculator evaluates through an
AST whitelist (`safe_eval`), rejecting attribute access, calls other than allowed
functions, and dunder tricks. Tool results are truncated to 2000 chars before
re-entering the context.

**14. Why run tools in an executor instead of awaiting them directly?**
The tools do blocking `urllib`/`requests` I/O; calling them on the event loop would
freeze SSE streaming and the keep-alive heartbeat for seconds.
`asyncio.get_event_loop().run_in_executor(None, functools.partial(...))` offloads
them to a worker thread.

**15. A model forgets `FINAL:` and just answers — what happens?**
`_parse_action` returns `None`, so the reply is treated as a final answer (bare
replies are streamed as plain text and emitted as `final`). This "be liberal in what
you accept" choice stops the loop from wasting iterations or erroring.

**16. Why do reasoning models like gpt-oss sometimes skip the tool despite instructions, and how does the app cope?**
Reasoning models tend to compute internally and answer directly. The loop never
*requires* a tool: any non-ACTION reply ends the turn as a final answer, so the
response is still correct (maybe less verifiable). Models that follow protocol
better (e.g. qwen3.8) execute the loop faithfully.

---

## C. Streaming & Frontend

**17. List the SSE event types and their payloads.**
`status` (progress text like "Thinking… 5s"), `tool_start` (`tool`, `label`, `args`),
`tool_end` (`tool`, `label`, `summary`), `token` (partial `text`), `final` (complete
`text`), `error` (friendly `text`). The server writes each as `data: {json}\n\n`.

**18. How does the "protocol leak" problem get solved while still streaming tokens?**
Each model reply starts in `detecting` mode: leading text is buffered until either
the `^(FINAL|ACTION)\s*:\s*` keyword regex matches (→ `protocol` mode, never shown
raw) or 16 chars / a newline accumulate (→ `plain` mode, buffer flushed as a token).
`FINAL:` prefixes are stripped before the `final` event, and final answers already
fully streamed aren't re-emitted.

**19. Why is a heartbeat needed and how is it implemented without blocking?**
Between the request and the first token (or during long reasoning) nothing flows and
proxies/users may think it's dead. The loop awaits the model's `__anext__` wrapped in
a task with `asyncio.wait(timeout=2.0)`; on timeout it yields an elapsed-seconds
`status` event and keeps waiting. The pending task is cancelled in a `finally`.

**20. How does script.js decide the API base URL?**
Same-origin when served on port 8000 (FastAPI serves the UI itself); `file://` and
non-8000 origins fall back to `http://localhost:8000`; a `?api=` query param
overrides both. Sessions get an ID from `crypto.randomUUID` stored in
`sessionStorage`; "+ Nouveau chat" rotates it, giving server-side memory a fresh key.

**21. What does the provider badge show and where does the data come from?**
On load, the UI GETs `/health` and renders `provider_label · model` (e.g. "Groq
Cloud · qwen/qwen3.8-27b") in the header; on failure it shows "API hors ligne". This
makes the active backend visible without opening devtools.

---

## D. API Hardening

**22. How does rate limiting work?**
`SlidingWindowLimiter` keeps a deque of `time.monotonic()` hit timestamps per
session_id; each request pops entries older than 60s and rejects with 429 +
retry-after when the count reaches `RATE_LIMIT_PER_MINUTE` (default 20). Monotonic
clock avoids wall-clock jumps. In-memory = per-process only.

**23. Why `time.monotonic()` rather than `time.time()` (or mixing `perf_counter`)?**
Monotonic ignores system clock adjustments (NTP, DST); mixing it with
`perf_counter` caused a real bug here (durations like "9394s"), so the codebase
standardized on monotonic for all elapsed-time math.

**24. Which validations does `ChatRequest` enforce?**
Message 1–`MAX_MESSAGE_CHARS` (4000) chars; session_id 1–64 chars matching
`^[A-Za-z0-9_\-]+$`; enforced by pydantic at parse time → 422 before any agent code
runs.

**25. How do the endpoints fail when unconfigured, and what text do they use?**
Both chat endpoints check `settings.is_configured` first and return 503 naming the
exact missing env var (`missing_key_env`, e.g. "GROQ_API_KEY is missing") — the
message adapts to the selected provider. Inside a stream, configuration and other
errors surface as an `error` SSE event instead of an HTTP error.

**26. Why serve the frontend from FastAPI (`StaticFiles` mount) instead of a separate static server?**
A separate `python -m http.server` caused cross-origin POSTs that the static server
answered with 501. Same-origin mounting removes CORS friction entirely, keeps
`API_BASE` empty, and the mount is registered last so API routes take precedence.

**27. What does `/health` report and when is it `degraded`?**
`status` (`ok`/`degraded`), resolved `model`, `provider`, `provider_label`.
`degraded` means the selected provider's key is missing — liveness works, chat will
503.

---

## E. Errors, Config & Testing

**28. What makes `_friendly_error` provider-aware? Give two examples.**
It branches on `provider_id`: Gemini 429s distinguish *daily* quota ("PerDay") and
suggest switching provider; non-Gemini 429s report per-minute free-tier limits;
401/invalid-key errors name the exact env var to fix via `missing_key_env`.

**29. Why are heavy imports lazy inside builder functions?**
So the API and test suite boot (and `/health` answers) without langgraph /
langchain-google-genai installed — the machine's pip mirror lacks them. A missing
stack surfaces as a readable `AgentNotConfigured` on first chat, not a crash at
import time.

**30. How do tests fake the LLM without network or keys?**
`FakeLLM` records every prompt and returns scripted replies, streaming them in
6-char chunks through `astream` to exercise the buffering state machine. Tests
monkeypatch the module-level `_agent`/`_mode` singletons, and a `clean_agent_state`
autouse fixture resets globals between tests.

**31. How is `Settings` tested without picking up the developer's real `.env`?**
A helper constructs `Settings(_env_file=None, **kwargs)` passing env-var-style
aliases (`LLM_PROVIDER="groq"`, `GROQ_API_KEY=...`) — hermetic and identical to how
production values arrive. Gotcha: without `populate_by_name`, field-name kwargs are
silently ignored; you must use the aliases.

**32. What do the provider-specific builder tests assert?**
That groq builds tier-2 `openai_compat` (never the langgraph tier) targeting
`api.groq.com`, with no `reasoning_effort` in `model_kwargs`; that gemini keeps it;
and that an unconfigured provider raises `AgentNotConfigured` naming the right env
var (e.g. `GROQ_API_KEY`).

**33. Current suite status and what it covers?**
62 passed. Calculator (incl. injection rejection), rate limiter, provider registry &
selection, per-provider agent construction, `/health`, both chat endpoints with a
stubbed agent, SSE event ordering, and the text-protocol loop (direct answers, tool
calls, unknown tools, bare replies, history growth, error mapping).

---

## F. Ops & Debugging Scenarios

**34. Uvicorn fails with WinError 10048 / 10013 on Windows. What's happening?**
10048: the port is already bound — another (possibly forgotten) server instance is
running; find it via `netstat -ano | findstr :8000` and kill the PID, or use another
port. 10013: Windows reserved/excluded port range — pick a different port (e.g.
8010).

**35. The server starts but says "GOOGLE_API_KEY is not set" while you intend to use Groq.**
`LLM_PROVIDER` still says `gemini` in `.env`, so the app checks the Google key
(now empty). Flip `LLM_PROVIDER=groq` and restart — `is_configured` then checks
`GROQ_API_KEY`. `/health` shows which provider is live.

**36. Every request to Groq returns 404 "model not found". Diagnosis?**
Model catalogs drift — e.g. Groq retired `llama-3.3-70b-versatile`. Auth would fail
with 401 if the key were bad, so 404 means the model name. List the account's models
(`GET /openai/v1/models`) and set `MODEL_NAME` (the app default was moved to
`qwen/qwen3.8-27b` for exactly this reason).

**37. Chat answers arrive but tool calls never trigger on gpt-oss-120b. Why, and what's a fix?**
That model reasons internally and tends to answer directly, skipping the text
protocol. The loop handles it gracefully as a direct answer. Fix: prefer a
protocol-obedient model (e.g. qwen3.8), or strengthen the system prompt / raise
`AGENT_MAX_ITERATIONS`.

**38. Why install with `pip --no-cache-dir` on this machine?**
The sandbox has limited RAM; pip's cache step can trigger MemoryError during large
installs. `--no-cache-dir` skips writing wheels to cache, and
`requirements-sandbox.txt` pins mirror-compatible versions (langchain 0.2.17,
langchain-openai 0.1.25, duckduckgo_search instead of the unavailable ddgs/langgraph).

**39. The UI loads but shows "API hors ligne" in the badge. First three checks?**
(1) Is the backend actually up: `curl http://127.0.0.1:8000/health`. (2) Is the UI
served from the right origin — open http://localhost:8000 (FastAPI) rather than a
static server on :5500, or pass `?api=`. (3) Console/network tab: CORS or connection
refused will show there.
