"""
retrieval/search.py — Busca semântica no ChromaDB

Dado uma pergunta do usuário:
    1. Transforma a pergunta em embedding (mesmo modelo usado na indexação)
    2. Busca os chunks mais similares no ChromaDB por similaridade de cosseno
    3. Aplica threshold mínimo de similaridade — chunks muito distantes são descartados
    4. Retorna os chunks encontrados com seus metadados (doc, página, score)

─── Histórico de ajustes de parâmetros ───────────────────────────────────────

SIMILARITY_THRESHOLD:
    Tentativa 1 — 0.40: ponto de partida recomendado. Funcionou bem na maioria
        das perguntas, mas uma pergunta fora do escopo retornou um chunk com
        score 0.45, ou seja, passou pelo filtro e foi enviada à LLM.
    Tentativa 2 — 0.90: subimos para evitar respostas irrelevantes. Funcionou
        para perguntas simples, mas causou um bug crítico: perguntas que cobrem
        dois ou mais PDFs não retornavam nenhum chunk, porque os chunks de cada
        documento individualmente ficavam abaixo de 0.90. A LLM respondia
        "fora do escopo" mesmo tendo a informação nos documentos.
    Valor atual — 0.55: compromisso entre os dois extremos. O threshold de
        cosseno é apenas um pré-filtro grosseiro; a filtragem fina ficou por
        conta do cross-encoder (reranker), que descarta chunks com score <= 0
        e é muito mais preciso para julgar relevância.

TOP_K:
    Tentativa 1 — 7: suficiente para perguntas de documento único.
    Valor atual — 12: aumentado junto com a redução do threshold para garantir
        que perguntas multi-documento pesquem candidatos de 2+ PDFs antes do
        reranking reduzir para os 3 melhores.

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quantos chunks retornar na busca inicial — aumentado para cobrir perguntas
# que abrangem múltiplos documentos (mais candidatos antes do reranking)
TOP_K = 12

# Score mínimo de similaridade de cosseno — serve apenas como pré-filtro grosseiro
# para descartar chunks completamente aleatórios antes do cross-encoder.
# O reranker (cross-encoder) é quem faz a filtragem fina: já descarta tudo com
# score <= 0, que é um guardião muito mais preciso do que este threshold.
# 0.90 era alto demais: descartava chunks relevantes de documentos secundários
# em perguntas que abrangem 2+ PDFs.
SIMILARITY_THRESHOLD = 0.55

@dataclass
class ResultadoBusca:
    """
    Representa um chunk retornado pela busca semântica.

    Campos:
        doc:        nome do documento de origem
        pagina:     número da página de origem
        texto:      conteúdo do chunk
        score:      similaridade de cosseno com a pergunta (0 a 1)
        tipo:       "texto" para parágrafos, "tabela" para dados tabulares
    """
    doc   : str
    pagina: int
    texto : str
    score : float
    tipo  : str = "texto"


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
                tipo  = meta.get("tipo", "texto"),
            ))

        return chunks


if __name__ == "__main__":
    # Teste rápido da busca
    # Bug corrigido: o bloco de impressão dos resultados estava num `else` do
    # for externo (for-else do Python), não do if interno. Isso fazia o último
    # resultado ser impresso duas vezes após o loop terminar. Corrigido movendo
    # a impressão para dentro do else correto (o do `if not resultados`).
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
                print(f"    {r.texto[:200]}...")
                print()