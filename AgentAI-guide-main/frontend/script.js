// script.js
document.addEventListener("DOMContentLoaded", () => {

    const chatBox   = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn   = document.getElementById("send-btn");
    const statusEl  = document.getElementById("status-label");

    // URL de l'API backend
    const API_URL = "http://127.0.0.1:8000/agent-chat";

    // ── État ──────────────────────────────────────────────
    let isThinking = false;

    // ── Statut header ─────────────────────────────────────
    function setStatus(state) {
        const parent = statusEl?.parentElement;
        if (!parent || !statusEl) return;
        const labels = { online: "En ligne", thinking: "Réflexion…", error: "Hors ligne" };
        parent.className = "header-status" + (state !== "online" ? ` ${state}` : "");
        statusEl.textContent = labels[state] ?? "En ligne";
    }

    // ── Canvas background ─────────────────────────────────
    const canvas = document.getElementById("bg-canvas");
    if (canvas) {
        const ctx = canvas.getContext("2d");

        function resizeCanvas() {
            canvas.width  = window.innerWidth;
            canvas.height = window.innerHeight;
        }
        resizeCanvas();
        window.addEventListener("resize", resizeCanvas);

        class Particle {
            constructor() { this.reset(true); }
            reset(initial = false) {
                this.x     = Math.random() * canvas.width;
                this.y     = initial ? Math.random() * canvas.height : canvas.height + 10;
                this.vx    = (Math.random() - .5) * .28;
                this.vy    = -(Math.random() * .4 + .1);
                this.r     = Math.random() * 1.5 + .4;
                this.alpha = Math.random() * .5 + .1;
                this.color = Math.random() > .5 ? "168,85,247" : "124,58,237";
                this.phase = Math.random() * Math.PI * 2;
                this.speed = Math.random() * .018 + .005;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;
                this.phase += this.speed;
                this.a = this.alpha * (.55 + .45 * Math.sin(this.phase));
                if (this.y < -10) this.reset(false);
            }
            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.r, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(${this.color},${this.a})`;
                ctx.fill();
            }
        }

        const particles = Array.from({ length: 50 }, () => new Particle());

        function drawLinks() {
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const d  = Math.sqrt(dx*dx + dy*dy);
                    if (d < 100) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(168,85,247,${(1 - d/100) * .055})`;
                        ctx.lineWidth = .5;
                        ctx.stroke();
                    }
                }
            }
        }

        (function loop() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawLinks();
            particles.forEach(p => { p.update(); p.draw(); });
            requestAnimationFrame(loop);
        })();
    }

    // ── Scroll ────────────────────────────────────────────
    function scrollToBottom() {
        chatBox.scrollTo({ top: chatBox.scrollHeight, behavior: "smooth" });
    }

    // ── Ajouter un message ────────────────────────────────
    function addMessage(text, sender, isHTML = false) {
        // Supprimer écran de bienvenue au 1er message
        chatBox.querySelector(".welcome-screen")?.remove();

        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);

        if (isHTML) {
            messageElement.innerHTML = text;
        } else {
            // Affichage texte sécurisé avec sauts de ligne
            text.split("\n").forEach((line, i, arr) => {
                messageElement.appendChild(document.createTextNode(line));
                if (i < arr.length - 1) messageElement.appendChild(document.createElement("br"));
            });
        }

        chatBox.appendChild(messageElement);
        scrollToBottom();
    }

    function addError(message) {
        chatBox.querySelector(".welcome-screen")?.remove();
        const el = document.createElement("div");
        el.classList.add("error-message");
        el.textContent = message;
        chatBox.appendChild(el);
        scrollToBottom();
    }

    // ── Typing indicator ──────────────────────────────────
    function showTypingIndicator() {
        if (document.querySelector(".typing-indicator")) return;
        chatBox.querySelector(".welcome-screen")?.remove();
        const typingEl = document.createElement("div");
        typingEl.classList.add("typing-indicator");
        typingEl.innerHTML = "<span></span><span></span><span></span>";
        chatBox.appendChild(typingEl);
        scrollToBottom();
    }

    function hideTypingIndicator() {
        document.querySelector(".typing-indicator")?.remove();
    }

    // ── Écran de bienvenue ────────────────────────────────
    function renderWelcome() {
        chatBox.innerHTML = `
            <div class="welcome-screen">
                <h2 class="welcome-headline">Bonjour,<br>comment puis-je vous aider ?</h2>
                <p class="welcome-sub">Agent multi-outils alimenté par IA. Posez-moi n'importe quelle question.</p>
                <div class="welcome-chips">
                    <button class="chip" data-q="Quelles sont les dernières actualités ?">🌍 Actualités</button>
                    <button class="chip" data-q="Calcule 15% de 840 €">🔢 Calcul rapide</button>
                    <button class="chip" data-q="Explique le machine learning en 3 points">🤖 Machine learning</button>
                    <button class="chip" data-q="Quel LLM choisir pour mon projet ?">💡 Choisir un LLM</button>
                </div>
            </div>`;

        chatBox.querySelectorAll(".chip").forEach(chip => {
            chip.addEventListener("click", () => {
                userInput.value = chip.dataset.q;
                sendMessage();
            });
        });
    }
    renderWelcome();

    // ── Envoi du message ──────────────────────────────────
    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (!messageText || isThinking) return;

        addMessage(messageText, "user");
        userInput.value = "";
        isThinking = true;
        sendBtn.disabled = true;
        setStatus("thinking");
        showTypingIndicator();

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: messageText }),
            });

            hideTypingIndicator();

            if (!response.ok) throw new Error(`Erreur serveur (${response.status})`);

            const data = await response.json();

            if (data.error) {
                addError(`Erreur : ${data.error}`);
            } else {
                const botResponse = data.response;
                const isHTML = /<[a-z][\s\S]*>/i.test(botResponse);
                addMessage(botResponse, "bot", isHTML);
            }

            setStatus("online");

        } catch (error) {
            hideTypingIndicator();
            addError(`Impossible de contacter l'API : ${error.message}`);
            setStatus("error");
        } finally {
            isThinking = false;
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    // ── Événements ────────────────────────────────────────
    sendBtn.addEventListener("click", sendMessage);

    userInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    userInput.focus();
});
