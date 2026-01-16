from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Importer seulement l'exécuteur de l'agent et le llm
from agent import agent_executor, llm

# --- Application FastAPI ---
app = FastAPI(title="API pour Agent LLM Flexible", version="5.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AgentRequest(BaseModel):
    message: str

@app.post("/agent-chat")
async def agent_chat(request: AgentRequest):
    if not agent_executor or not llm:
        raise HTTPException(status_code=500, detail="Agent ou LLM non configuré.")
    
    try:
        # Invoquer l'agent avec la question de l'utilisateur
        response_raw = agent_executor.invoke({"input": request.message})
        agent_response = response_raw['output']
        
        # Retourner la réponse telle qu'elle est
        return {"response": agent_response}

    except Exception as e:
        # Gérer les erreurs
        error_message = f"Erreur lors de l'exécution : {str(e)}"
        print(error_message)
        return {"response": f"Désolé, une erreur est survenue. Veuillez réessayer."}

@app.get("/")
def read_root():
    return {"status": "L'API de l'agent LLM flexible est en ligne !"}
