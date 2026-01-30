# 🤖 Smart Agent API

> An intelligent conversational agent powered by Google Gemini, equipped with web search, Wikipedia, and calculation capabilities.

## ✨ What It Does

This agent can:
- 🔍 Search the web for real-time information
- 📚 Query Wikipedia for detailed knowledge
- 🧮 Perform complex mathematical calculations
- 💬 Have natural conversations and answer questions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Your API Key
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 3. Run the Backend
```bash
uvicorn main:app --reload
```

### 4. Open the Frontend
Simply open `index.html` in your browser, and start chatting!

## 🛠️ Tech Stack

- **Backend**: FastAPI + LangChain + Google Gemini
- **Frontend**: HTML, CSS, JavaScript
- **Tools**: DuckDuckGo Search, Wikipedia, Math Calculator


## 💡 Example Queries

- "What's the latest news about AI?"
- "Tell me about Albert Einstein"
- "Calculate the square root of 144"
- "What's the weather like today?"

## 🤝 Contributing

Feel free to fork, improve, and submit pull requests!

---

Built with ❤️ using LangChain and Google Gemini
