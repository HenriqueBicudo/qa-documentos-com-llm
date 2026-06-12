"""
api/chat.py — API REST do sistema RAG via FastAPI

Expõe um único endpoint POST /chat que recebe uma pergunta e retorna
duas saídas obrigatórias (exigência do professor):
    - debug:    prompt montado + chunks usados (para avaliação)
    - resposta: texto final gerado pela LLM

Execute:
    uvicorn api.chat:app

Nota sobre GPU: o CUDA_VISIBLE_DEVICES="" original foi removido.
Era necessário em GPUs com pouca VRAM para forçar os modelos de embedding
e reranker na CPU, liberando VRAM para o Ollama. Com 8GB de VRAM (RTX 4060 Ti)
todos os modelos cabem na GPU simultaneamente:
    e5-large:       ~1.1 GB
    cross-encoder:  ~0.2 GB
    qwen3:8b:       ~5.2 GB
    ─────────────────────────
    Total:          ~6.5 GB → cabe nos 8 GB disponíveis
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
    tipo  : str = "texto"


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
        resposta:      texto final gerado pelo qwen3:8b
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
                tipo   = c.tipo,
            )
            for c in resultado.chunks_usados
        ],
        prompt_final  = resultado.prompt_final,
        resposta      = resultado.resposta,
    )