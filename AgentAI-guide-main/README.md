# Agent LLM Web App - Guide

## 1. Introduction

Welcome! This project is a simple yet powerful web application that allows you to chat with an advanced AI Agent. This agent isn't just a standard chatbot; it's powered by Google's Gemini Pro model and equipped with a set of tools (Search, Wikipedia, Calculator) that allow it to browse the web, look up facts, and perform calculations to answer your questions accurately.

The project is built with a **FastAPI backend** (Python) that serves the AI logic and a **vanilla HTML, CSS, and JavaScript frontend** for the user interface.

### Technologies Used
- **Backend:** Python, FastAPI, LangChain, Google Gemini
- **Frontend:** HTML, CSS, JavaScript
- **Core Logic:** A "ReAct" (Reasoning and Acting) Agent that can decide which tool to use based on the user's query.

## 2. Project Architecture

The application is divided into two main parts:

1.  **Frontend (Client-Side):** The `index.html`, `style.css`, and `script.js` files create the chat interface you see in your browser. When you send a message, the JavaScript sends a request to our backend API.
2.  **Backend (Server-Side):** The `main.py` file, running with FastAPI, receives requests from the frontend. It passes your message to the LangChain Agent, which uses the Gemini LLM to reason and utilize its tools. Once an answer is found, the backend sends it back to the frontend to be displayed.

## 3. Features

- **Interactive Chat UI:** A clean and simple web interface for conversation.
- **Multi-Tool Agent:** The AI can:
    - **Search the web** for up-to-date information.
    - **Query Wikipedia** for factual data.
    - **Perform math calculations**.
- **Real-time Feedback:** A "typing indicator" shows when the agent is processing a request.
- **Decoupled Architecture:** The frontend and backend are separate, which is a modern standard in web development.

## 4. Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

- **Python 3.8+:** Make sure you have Python installed. You can check with `python --version`.
- **A Code Editor:** VS Code, PyCharm, or any other editor of your choice.
- **Google API Key:** You need an API key for the Gemini model. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### Step-by-Step Guide

**Step 1: Clone the Repository**
```
git clone https://github.com/zeynbabid246-beep/AgentAI-LLM
cd AgentAI-guide
```

Step 2: Create a Python Virtual Environment

#### For Windows
```
python -m venv venv
.\venv\Scripts\activate
```
#### For macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```
Step 3: Install Dependencies
```
pip install -r requirements.txt
```
Step 4: Create the .env File
This file will store your secret API key. Create a file named .env in the root of your project directory and add your Google API key to it.
```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```
Important: Never share this file or commit it to version control.
Step 5: Run the Backend Server
Use uvicorn to start the FastAPI server.
```
uvicorn main:app --reload
```
- `--reload` makes the server restart automatically after you make code changes.
- The server will be running at `http://127.0.0.1:8000`.

**Step 6: Open the Frontend**
Navigate to your project folder and open the `index.html` file in your web browser. You can usually do this by simply double-clicking the file.

You are now ready to chat with your AI agent!

## 5. How It Works: Code Breakdown

### `main.py` (Backend)
- **FastAPI Setup:** Sets up a web server with an endpoint at `/agent-chat` that accepts POST requests.
- **LLM Initialization:** It loads your `GOOGLE_API_KEY` and initializes the `ChatGoogleGenerativeAI` model.
- **Tool Creation:** Three tools are defined: `DuckDuckGoSearchRun`, `WikipediaQueryRun`, and `LLMMathChain`. Each is wrapped in a `Tool` object with a name and a description. The description is crucial, as it tells the agent *when* to use the tool.
- **Agent and Executor:** `create_react_agent` assembles the LLM, the tools, and a prompt template. The `AgentExecutor` is the runtime that actually executes the agent's decisions (e.g., "Use the Calculator tool with the input '5+5'").
- **API Endpoint Logic:** The `/agent-chat` function takes the user's message, passes it to the `agent_executor.invoke`, and returns the final `output` as a JSON response.

### `script.js` (Frontend Logic)
- **DOM Event Listeners:** It waits for the user to click the "Send" button or press "Enter".
- **`sendMessage()` Function:**
    1. It takes the user's text and displays it in the chat box.
    2. It shows a "typing indicator" to provide feedback.
    3. It uses the `fetch()` API to send the message to the backend at `http://127.0.0.1:8000/agent-chat`.
    4. When the backend responds, it hides the typing indicator and displays the bot's message in the chat.
    5. It includes error handling in case the API cannot be reached.
