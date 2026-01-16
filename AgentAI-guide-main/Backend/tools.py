from langchain.agents import Tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.chains import LLMMathChain
from langchain_google_genai import ChatGoogleGenerativeAI

def get_tools(llm: ChatGoogleGenerativeAI) -> list:
    """
    Initialise et retourne une liste d'outils pour l'agent.
    Prend le modèle de langage (llm) en argument car certains outils en ont besoin.
    """
    
    # 1. Outil de recherche Web
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Très utile pour répondre aux questions sur l'actualité, les événements récents et les sujets généraux."
    )

    # 2. Outil Wikipedia
    wikipedia_api = WikipediaAPIWrapper()
    wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api)
    wikipedia_tool = Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Utile pour rechercher des informations factuelles approfondies sur des sujets, des personnes et des lieux dans une encyclopédie."
    )

    # 3. Outil de calcul (a besoin du llm pour fonctionner)
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    calculator_tool = Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Indispensable pour répondre aux questions nécessitant un calcul mathématique précis."
    )

    # Regrouper les outils dans une liste
    tools = [search_tool, wikipedia_tool, calculator_tool]
    return tools

