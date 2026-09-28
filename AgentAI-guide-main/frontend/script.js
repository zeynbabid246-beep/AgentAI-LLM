// ============================================================
//  Smart Agent — script.js
//  Streams replies from the FastAPI backend over SSE.
//
//  POST /chat/stream   { message, session_id }
//  SSE events: {type: status | tool_start | tool_end | token | final | error}
// ============================================================

console.log("[Smart Agent] UI v3 (status events + streaming)");

// Resolve the backend base URL.
// - Served by FastAPI itself (localhost:8000): same-origin, no prefix needed.
// - Served elsewhere (static server on :5500, file://, LAN IP...): fall back
//   to localhost:8000, or ?api=http://host:port to point anywhere.
const API_BASE = (() => {
  if (location.origin.startsWith("http") && location.port === "8000") return "";
  if (location.protocol === "file:") return "http://localhost:8000";
  const override = new URLSearchParams(location.search).get("api");
  return override || "http://localhost:8000";
})();
const STREAM_URL = `${API_BASE}/chat/stream`;
const REQUEST_TIMEOUT_MS = 120000;

const textarea = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");

// ---- Session management (server-side memory key) ----
let sessionId = sessionStorage.getItem("agentSessionId");
if (!sessionId) {
  sessionId =
    crypto.randomUUID?.().replace(/-/g, "").slice(0, 24) ||
    `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  sessionStorage.setItem("agentSessionId", sessionId);
}

function startNewChat() {
  sessionId =
    crypto.randomUUID?.().replace(/-/g, "").slice(0, 24) ||
    `s-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  sessionStorage.setItem("agentSessionId", sessionId);
  document.getElementById("chatMessages").innerHTML = "";
  appendMessage("bot", "New conversation started. How can I help you?");
}

// ---- Auto-resize textarea ----
textarea.addEventListener("input", () => {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + "px";
});

// Enter to send (Shift+Enter for newline)
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ---- Main send function: POST + SSE reader ----
let inFlight = null; // AbortController for the current request

async function sendMessage() {
  const input = textarea.value.trim();
  if (!input) return;

  textarea.value = "";
  textarea.style.height = "auto";

  appendMessage("user", input);

  sendBtn.disabled = true;
  const botRow = appendStreamingBotMessage();
  const statusEl = botRow.querySelector(".stream-status");

  const controller = new AbortController();
  inFlight = controller;
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let fullText = "";

  try {
    const response = await fetch(STREAM_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input, session_id: sessionId }),
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const err = await response.json();
        if (err.detail) detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
      } catch (_) { /* keep status-line detail */ }
      throw new Error(detail);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const rawEvent = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const dataLine = rawEvent.split("\n").find((l) => l.startsWith("data: "));
        if (!dataLine) continue;

        let event;
        try {
          event = JSON.parse(dataLine.slice(6));
        } catch (_) {
          continue;
        }
        handleAgentEvent(event);
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      appendErrorToBubble(statusEl, "⏱️ The request timed out. Please try again.");
    } else {
      appendErrorToBubble(statusEl, `⚠️ ${err.message || "Connection error. Is the backend running?"}`);
    }
  } finally {
    clearTimeout(timeout);
    inFlight = null;
    sendBtn.disabled = false;
    textarea.focus();
  }

  // ---- Single SSE event handler ----
  function handleAgentEvent(event) {
    switch (event.type) {
      case "status": {
        // Keep-alive progress line from the backend ("Thinking… 4s",
        // "Searching the web… done, thinking"). Never clears the text bubble.
        if (statusEl) statusEl.textContent = event.text || "";
        break;
      }
      case "tool_start": {
        const chip = addToolChip(botRow, event.label || event.tool);
        statusEl.textContent = `${event.label || event.tool}…`;
        chip.dataset.tool = event.tool;
        break;
      }
      case "tool_end": {
        completeToolChip(botRow, event.tool);
        statusEl.textContent = "";
        break;
      }
      case "token": {
        fullText += event.text;
        renderIntoBubble(botRow, fullText);
        scrollToBottom();
        break;
      }
      case "final": {
        fullText = event.text || fullText;
        renderIntoBubble(botRow, fullText);
        finishStreaming(botRow);
        scrollToBottom();
        break;
      }
      case "error": {
        appendErrorToBubble(statusEl, `⚠️ ${event.text}`);
        finishStreaming(botRow);
        break;
      }
    }
  }
}

// ---- Create the bot bubble once, stream tokens into it ----
function appendStreamingBotMessage() {
  const chatMessages = document.getElementById("chatMessages");

  const row = document.createElement("div");
  row.className = "message-row bot-row streaming";

  const avatar = document.createElement("div");
  avatar.className = "avatar bot-avatar";
  avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" fill="currentColor"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;

  const bubble = document.createElement("div");
  bubble.className = "message bot-message";

  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = "Agent LLM";
  bubble.appendChild(label);

  const status = document.createElement("div");
  status.className = "stream-status";
  status.textContent = "Thinking…";
  bubble.appendChild(status);

  const textNode = document.createElement("span");
  textNode.className = "message-text";
  bubble.appendChild(textNode);

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollToBottom();
  return row;
}

function renderIntoBubble(botRow, text) {
  const statusEl = botRow.querySelector(".stream-status");
  if (statusEl.textContent === "Thinking…") statusEl.textContent = "";
  const textNode = botRow.querySelector(".message-text");
  textNode.innerHTML = formatText(text);
}

function finishStreaming(botRow) {
  botRow.classList.remove("streaming");
  const statusEl = botRow.querySelector(".stream-status");
  if (statusEl) statusEl.textContent = "";
  completeToolChip(botRow, null); // mark all remaining chips done
}

function appendErrorToBubble(statusEl, message) {
  if (statusEl) statusEl.innerHTML = `<span class="stream-error">${message}</span>`;
}

// ---- Tool-step chips ----
function addToolChip(botRow, labelText) {
  const bubble = botRow.querySelector(".bot-message");
  let chips = bubble.querySelector(".tool-chips");
  if (!chips) {
    chips = document.createElement("div");
    chips.className = "tool-chips";
    bubble.insertBefore(chips, bubble.querySelector(".message-text"));
  }
  const chip = document.createElement("span");
  chip.className = "tool-chip running";
  chip.textContent = `⚙️ ${labelText}…`;
  chips.appendChild(chip);
  scrollToBottom();
  return chip;
}

function completeToolChip(botRow, toolName) {
  const chips = botRow.querySelectorAll(".tool-chip.running");
  if (toolName === null) {
    chips.forEach((c) => {
      c.classList.remove("running");
      c.textContent = c.textContent.replace(/…$/, " ✓");
    });
    return;
  }
  chips.forEach((c) => {
    if (c.dataset.tool === toolName) {
      c.classList.remove("running");
      c.textContent = c.textContent.replace(/…$/, " ✓");
    }
  });
}

// ---- Fallback plain message append (intro / new-chat notices) ----
function appendMessage(role, text) {
  const chatMessages = document.getElementById("chatMessages");

  const row = document.createElement("div");
  row.className = `message-row ${role === "user" ? "user-row" : "bot-row"}`;

  const avatar = document.createElement("div");
  avatar.className = `avatar ${role === "user" ? "user-avatar" : "bot-avatar"}`;
  avatar.innerHTML =
    role === "user"
      ? `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="1.5"/><path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" fill="currentColor"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;

  const bubble = document.createElement("div");
  bubble.className = `message ${role === "user" ? "user-message" : "bot-message"}`;

  if (role === "bot") {
    const label = document.createElement("span");
    label.className = "message-label";
    label.textContent = "Agent LLM";
    bubble.appendChild(label);
  }

  const textNode = document.createElement("span");
  textNode.innerHTML = formatText(text);
  bubble.appendChild(textNode);

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatMessages.appendChild(row);
  scrollToBottom();
}

// ---- Scroll to bottom ----
function scrollToBottom() {
  const chatMessages = document.getElementById("chatMessages");
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
}

// ---- Basic markdown rendering (escaped first) ----
function formatText(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code style='background:rgba(99,102,241,0.15);padding:2px 6px;border-radius:4px;font-size:12px;'>$1</code>")
    .replace(/\n/g, "<br>");
}

// ---- Provider badge: show which LLM provider/model the backend is running ----
async function loadProviderBadge() {
  const badge = document.getElementById("providerBadge");
  if (!badge) return;
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout?.(8000) });
    if (!resp.ok) throw new Error(`health ${resp.status}`);
    const data = await resp.json();
    const model = data.model || data.provider || "?";
    badge.textContent = `${data.provider_label || data.provider || "LLM"} · ${model}`;
    badge.title = badge.textContent;
    badge.classList.add("ok");
  } catch (_err) {
    badge.textContent = "API hors ligne";
  }
}
loadProviderBadge();

// ---- Wire header buttons ----
newChatBtn?.addEventListener("click", startNewChat);
