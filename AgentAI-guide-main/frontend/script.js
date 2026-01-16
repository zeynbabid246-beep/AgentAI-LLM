// script.js
document.addEventListener("DOMContentLoaded", () => {
    const chatBox = document.getElementById("chat-box");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");

    // URL de notre API backend
    const API_URL = "http://127.0.0.1:8000/agent-chat";

    // Fonction pour ajouter un message à l'interface
    function addMessage(text, sender) {
        const messageElement = document.createElement("div");
        messageElement.classList.add("message", `${sender}-message`);
        messageElement.textContent = text;
        chatBox.appendChild(messageElement);
        scrollToBottom();
    }    // Fonction pour gérer l'indicateur "typing"
    function showTypingIndicator() {
        // S'assurer qu'un seul indicateur est présent
        if (document.querySelector('.typing-indicator')) return;

        const typingElement = document.createElement("div");
        typingElement.classList.add("message", "bot-message", "typing-indicator");
        typingElement.innerHTML = '<span></span><span></span><span></span>';
        chatBox.appendChild(typingElement);
        scrollToBottom();
    }

    function hideTypingIndicator() {
        const typingElement = document.querySelector('.typing-indicator');
        if (typingElement) {
            chatBox.removeChild(typingElement);
        }
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Fonction pour envoyer le message au backend
    async function sendMessage() {
        const messageText = userInput.value.trim();
        if (messageText === "") return;

        // Afficher le message de l'utilisateur
        addMessage(messageText, "user");
        userInput.value = "";

        // Afficher l'indicateur "typing"
        showTypingIndicator();

        try {
            // Envoyer la requête à l'API
            const response = await fetch(API_URL, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    message: messageText,
                }),
            });

            // Cacher l'indicateur une fois la réponse reçue
            hideTypingIndicator();

            if (!response.ok) {
                throw new Error("Erreur réseau ou du serveur.");
            }

            const data = await response.json();

            if (data.error) {
                addMessage(`Erreur : ${data.error}`, "bot");
            } else {
                // Afficher la réponse du bot directement
                const botResponse = data.response;
                addMessage(botResponse, "bot");
            }

        } catch (error) {
            hideTypingIndicator();
            addMessage(`Impossible de contacter l'API : ${error.message}`, "bot");
        }
    }

    // Gérer l'envoi avec le bouton et la touche "Entrée"
    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            sendMessage();
        }
    });

    // Message de bienvenue
    addMessage("Bonjour ! Je suis un agent multi-outils. Posez-moi une question sur l'actualité, une information factuelle ou un calcul !", "bot");
});