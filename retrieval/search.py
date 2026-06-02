"""
retrieval/search.py — Busca semântica no ChromaDB

Dado uma pergunta do usuário:
    1. Transforma a pergunta em embedding (mesmo modelo usado na indexação)
    2. Busca os chunks mais similares no ChromaDB por similaridade de cosseno
    3. Aplica threshold mínimo de similaridade — chunks muito distantes são descartados
    4. Retorna os chunks encontrados com seus metadados (doc, página, score)

Execute para testar:
    python -m retrieval.search
"""

import torch
import chromadb

from pathlib               import Path
from dataclasses           import dataclass
from sentence_transformers import SentenceTransformer
from config.logger         import get_logger

logger = get_logger(__name__)

# ── Caminhos e configurações ──────────────────────────────────────────────────

CHROMA_DIR      = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "ipardes_rag"
MODELO_EMBEDDING = "intfloat/multilingual-e5-large"

# Mesma coisa do embed.py — usa GPU se disponível, senão CPU - Perguntar pro professor se GPU é permitida e se usamos CUDA (NVIDIA) ou ROCm (AMD)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quantos chunks retornar na busca inicial
TOP_K = 7

# Score mínimo de similaridade (0 a 1) — abaixo disso descarta o chunk
# 1.0 = idêntico, 0.0 = sem relação. 0.4 é um bom ponto de partida.
# Primeiro teste foi com 0.4 mas das 5 respostas 1 veio estranha, ent inseri 5 perguntas para ver se o resultado melhora, e se não melhorar podemos ajustar esse threshold.
# ATT: Em mais perguntas o Modelo performou bem, mas a pergunta sem relação ao assunto ele respondeu com um chunk que tinha score 0.45, ou seja, acima do threshold, então talvez seja necessário aumentar o threshold pra evitar esse tipo de resposta irrelevante.
SIMILARITY_THRESHOLD = 0.90

@dataclass
class ResultadoBusca:
    """
    Representa um chunk retornado pela busca semântica.

    Campos:
        doc:        nome do documento de origem
        pagina:     número da página de origem
        texto:      conteúdo do chunk
        score:      similaridade de cosseno com a pergunta (0 a 1)
    """
    doc   : str
    pagina: int
    texto : str
    score : float


class BuscadorRAG:
    """
    Realiza busca semântica no ChromaDB para recuperar chunks relevantes.

    O mesmo modelo de embedding usado na indexação é usado aqui para
    garantir que os vetores da pergunta e dos chunks estejam no mesmo
    espaço semântico.
    """

    def __init__(self):
        logger.info(f"Carregando modelo de embedding em {DEVICE}...")
        self.modelo = SentenceTransformer(MODELO_EMBEDDING, device=DEVICE)

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        logger.info(f"ChromaDB conectado — {self.collection.count()} chunks indexados")

    def buscar(self, pergunta: str, top_k: int = TOP_K) -> list[ResultadoBusca]:
        """
        Busca os chunks mais relevantes para uma pergunta.

        O prefixo "query: " é obrigatório para o modelo e5 — diferencia
        semanticamente perguntas de documentos durante a busca.

        Chunks com score abaixo do SIMILARITY_THRESHOLD são descartados —
        isso evita que o RAG use contexto irrelevante e alucine.
        """
        # Gera embedding da pergunta com prefixo obrigatório do e5
        embedding_pergunta = self.modelo.encode(
            f"query: {pergunta}",
            normalize_embeddings=True,
        ).tolist()

        # Busca no ChromaDB por similaridade de cosseno
        resultados = self.collection.query(
            query_embeddings=[embedding_pergunta],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        documentos = resultados["documents"][0]
        metadatas  = resultados["metadatas"][0]
        distancias = resultados["distances"][0]

        for texto, meta, distancia in zip(documentos, metadatas, distancias):
            # ChromaDB retorna distância de cosseno (0=igual, 2=oposto)
            # Convertemos para score de similaridade (1=igual, 0=sem relação)
            score = 1 - (distancia / 2)

            if score < SIMILARITY_THRESHOLD:
                logger.info(f"  Chunk descartado (score={score:.3f} < {SIMILARITY_THRESHOLD})")
                continue

            chunks.append(ResultadoBusca(
                doc   = meta["doc"],
                pagina= meta["pagina"],
                texto = texto,
                score = round(score, 4),
            ))

        return chunks


if __name__ == "__main__":
    # Teste rápido da busca
    buscador = BuscadorRAG()
    perguntas = [
        "Qual foi o crescimento do PIB do Paraná?",
        "Quais são as políticas públicas avaliadas no Brasil?",
        "Como está a expectativa de vida no Paraná?",
        "Qual a situação do emprego no Paraná em 2025?",
        "Qual é a capital da França?",  # fora do escopo — não deve retornar nada relevante
    ]
    for pergunta in perguntas:
        print(f"\n{'='*60}")
        print(f"Pergunta: {pergunta}")
        print('='*60)
        resultados = buscador.buscar(pergunta)

        if not resultados:
            print("Nenhum chunk relevante encontrado.")
        else:
            for i, r in enumerate(resultados, 1):
                print(f"[{i}] {r.doc} — página {r.pagina} — score {r.score}")
                print(f"    {r.texto[:150]}...")
    else:
        for i, r in enumerate(resultados, 1):
            print(f"[{i}] {r.doc} — página {r.pagina} — score {r.score}")
            print(f"    {r.texto[:200]}...")
            print()