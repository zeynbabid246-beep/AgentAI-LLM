import os
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate

from tools import get_tools

# Charger les variables d'environnement
load_dotenv()

# --- Initialisation du LLM ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, 
        convert_system_message_to_human=True
        )
except Exception as e:
    print(f"Erreur lors de la configuration de Google Gemini : {e}")
    llm = None

# --- Configuration du Prompt de l'Agent (simple et flexible) ---
if llm:
    tools = get_tools(llm)

    prompt_template = """
    Answer the following questions as best you can. You have access to the following tools:
    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: [Your final answer here - respond naturally and appropriately to the user's question]

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={"tools": tools, "tool_names": ", ".join([t.name for t in tools])}
    )

    # Créer l'agent
    agent = create_react_agent(llm, tools, prompt)

    # Créer l'exécuteur de l'agent
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )
else:
    agent_executor = None
