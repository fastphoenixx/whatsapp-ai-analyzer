import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ollama
from termcolor import colored

# Adiciona a raiz do projeto ao path para importar módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Importa o motor da Sprint 3
from src.llm.chat_engine import WhatsAppChat, OLLAMA_MODEL

app = FastAPI(
    title="WhatsApp AI Analyzer API",
    description="API para análise de conversas e chat RAG com DeepSeek R1",
    version="1.0.0"
)

# Configura CORS (Permite que qualquer frontend acesse)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta a pasta de relatórios para acesso via URL
REPORTS_DIR = Path("data/reports")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# --- ESTADO GLOBAL ---
# Carregamos o Chat Engine uma única vez na inicialização
print(colored("⏳ Inicializando Motor de IA para a API...", "yellow"))
chat_engine = None

@app.on_event("startup")
async def startup_event():
    global chat_engine
    try:
        # Inicializa conexão com Qdrant e Modelo de Embedding
        chat_engine = WhatsAppChat()
        print(colored("✅ API Pronta e Conectada à GPU!", "green"))
    except Exception as e:
        print(colored(f"❌ Falha crítica ao iniciar motor: {e}", "red"))

# --- MODELOS DE DADOS ---
class ChatRequest(BaseModel):
    message: str
    limit: int = 15

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "status": "online",
        "gpu": "AMD Radeon RX 6600 XT",
        "endpoints": ["/v1/chat", "/v1/reports/{filename}"]
    }

@app.post("/v1/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Recebe uma pergunta, busca contexto no Qdrant e gera resposta via DeepSeek.
    Retorna streaming de texto.
    """
    if not chat_engine:
        raise HTTPException(status_code=503, detail="Motor de IA não inicializado")

    # 1. Recuperação (RAG)
    context = chat_engine.get_context(req.message, limit=req.limit)
    
    # 2. Construção do Prompt
    system_prompt = f"""
    Você é um assistente de IA especialista neste grupo de WhatsApp.
    Responda em Português do Brasil.
    Use estritamente o contexto abaixo. Se não souber, diga que não sabe.
    
    CONTEXTO RECUPERADO:
    {context}
    """

    # 3. Gerador para Streaming
    async def generate():
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': req.message},
            ],
            stream=True,
        )
        
        for chunk in stream:
            yield chunk['message']['content']

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/v1/gallery")
async def list_reports():
    """Lista todos os gráficos gerados disponíveis"""
    if not REPORTS_DIR.exists():
        return []
    
    files = [f.name for f in REPORTS_DIR.glob("*.png")]
    return {
        "count": len(files),
        "files": files,
        "base_url": "/reports/"
    }

if __name__ == "__main__":
    import uvicorn
    print(colored("🚀 Iniciando Servidor Uvicorn...", "cyan"))
    uvicorn.run(app, host="0.0.0.0", port=8000)
