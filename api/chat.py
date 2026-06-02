"""
api/chat.py — API REST do sistema RAG via FastAPI

Expõe um único endpoint POST /chat que recebe uma pergunta e retorna
duas saídas obrigatórias (exigência do professor):
    - debug:    prompt montado + chunks usados (para avaliação)
    - resposta: texto final gerado pela LLM

Execute (CUDA_VISIBLE_DEVICES vazio pra deixar GPU livre pro Ollama):
    $env:CUDA_VISIBLE_DEVICES=""; uvicorn api.chat:app
"""

from fastapi         import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic        import BaseModel
from api.generator   import GeradorRAG
from config.logger   import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title       = "RAG IPARDES",
    description = "API de chat baseada em RAG sobre documentos do IPARDES",
    version     = "1.0.0",
)

# Permite requisições do Streamlit (mesmo host, porta diferente)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega o gerador uma vez ao iniciar — evita recarregar o modelo a cada requisição
gerador = GeradorRAG()


# ── Modelos de entrada/saída ───────────────────────────────────────────────────

class PerguntaRequest(BaseModel):
    pergunta: str


class ChunkInfo(BaseModel):
    doc   : str
    pagina: int
    score : float
    trecho: str


class ChatResponse(BaseModel):
    pergunta     : str
    chunks_usados: list[ChunkInfo]
    prompt_final : str
    resposta     : str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "mensagem": "API RAG IPARDES rodando"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: PerguntaRequest):
    """
    Recebe uma pergunta e retorna a resposta gerada via RAG.

    Saídas:
        chunks_usados: trechos recuperados do ChromaDB com doc, página e score
        prompt_final:  prompt completo enviado à LLM (para avaliação)
        resposta:      texto final gerado pelo Qwen2.5:7b-instruct
    """
    logger.info(f"POST /chat — pergunta: {request.pergunta}")
    resultado = gerador.responder(request.pergunta)

    return ChatResponse(
        pergunta      = resultado.pergunta,
        chunks_usados = [
            ChunkInfo(
                doc    = c.doc,
                pagina = c.pagina,
                score  = c.score,
                trecho = c.texto,
            )
            for c in resultado.chunks_usados
        ],
        prompt_final  = resultado.prompt_final,
        resposta      = resultado.resposta,
    )