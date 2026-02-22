// ============================================================
//  Mon Agent LLM — script.js
//  Backend: FastAPI @ localhost:8000/agent-chat
//  Payload:  POST { "message": "..." }
//  Response: { "response": "..." }
// ============================================================

const API_URL = "http://localhost:8000/agent-chat";

// Auto-resize textarea
const textarea = document.getElementById("userInput");
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

// ---- Main send function ----
async function sendMessage() {
  const input = textarea.value.trim();
  if (!input) return;

  textarea.value = "";
  textarea.style.height = "auto";

  appendMessage("user", input);

  const sendBtn = document.getElementById("sendBtn");
  sendBtn.disabled = true;

  const typingRow = showTypingIndicator();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input }),  // matches AgentRequest(message: str)
    });

    if (!response.ok) {
      throw new Error(`Erreur serveur : ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const reply = data.response || "Désolé, aucune réponse reçue.";

    typingRow.remove();
    appendMessage("bot", reply);

  } catch (err) {
    typingRow.remove();
    appendMessage("bot", `⚠️ ${err.message || "Erreur de connexion. Veuillez réessayer."}`);
    console.error("Erreur API:", err);
  } finally {
    sendBtn.disabled = false;
    textarea.focus();
  }
}

// ---- Append a chat message ----
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

// ---- Typing indicator ----
function showTypingIndicator() {
  const chatMessages = document.getElementById("chatMessages");

  const row = document.createElement("div");
  row.className = "message-row bot-row";

  const avatar = document.createElement("div");
  avatar.className = "avatar bot-avatar";
  avatar.innerHTML = `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" fill="currentColor"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.93 4.93l2.12 2.12M16.95 16.95l2.12 2.12M4.93 19.07l2.12-2.12M16.95 7.05l2.12-2.12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`;

  const typing = document.createElement("div");
  typing.className = "typing-indicator";
  typing.innerHTML = `<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>`;

  row.appendChild(avatar);
  row.appendChild(typing);
  chatMessages.appendChild(row);
  scrollToBottom();

  return row;
}

// ---- Scroll to bottom ----
function scrollToBottom() {
  const chatMessages = document.getElementById("chatMessages");
  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: "smooth" });
}

// ---- Basic markdown rendering ----
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
