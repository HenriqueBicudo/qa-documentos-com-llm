"""
retrieval/reranker.py — Reranking dos chunks recuperados via cross-encoder

O retrieval inicial (busca por similaridade de cosseno) é rápido mas
imperfeito — retorna chunks semanticamente próximos mas nem sempre os
mais relevantes para responder a pergunta.

O reranker resolve isso: recebe a pergunta + os chunks candidatos e
pontua cada par (pergunta, chunk) com um modelo cross-encoder, que
analisa os dois textos juntos (não separadamente como o bi-encoder).

Modelo escolhido: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    - Treinado especificamente para reranking multilíngue (mMARCO)
    - Suporta português nativamente
    - Leve (~120MB) — roda rápido mesmo na CPU
    - Mencionado diretamente no anexo do professor

Fluxo:
    1. BuscadorRAG retorna top-12 chunks por similaridade de cosseno
    2. RerankadorRAG repontua todos os 12 com o cross-encoder
    3. Retorna os top-3 mais relevantes segundo o cross-encoder
    4. Esses 3 vão pro prompt da LLM

Isso melhora a qualidade especialmente em perguntas que exigem
raciocínio mais preciso sobre o conteúdo dos chunks.
"""

from sentence_transformers  import CrossEncoder
from retrieval.search       import ResultadoBusca
from config.logger          import get_logger

logger = get_logger(__name__)

# ── Configurações ─────────────────────────────────────────────────────────────

MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Quantos chunks manter após o reranking
# Reduzimos de 7 para 3 — só os mais relevantes vão pro prompt da LLM
TOP_K_RERANKED = 3


class RerankadorRAG:
    """
    Reordena os chunks recuperados pelo BuscadorRAG usando um cross-encoder.

    O cross-encoder avalia cada par (pergunta, chunk) conjuntamente,
    capturando nuances que o bi-encoder de similaridade não consegue.
    """

    def __init__(self):
        logger.info(f"Carregando cross-encoder: {MODELO_RERANKER}...")
        # O cross-encoder roda bem na CPU — é leve e rápido
        self.modelo = CrossEncoder(MODELO_RERANKER)
        logger.info("Cross-encoder carregado.")

    def rerankear(
        self,
        pergunta: str,
        chunks  : list[ResultadoBusca],
        top_k   : int = TOP_K_RERANKED,
    ) -> list[ResultadoBusca]:
        """
        Reordena os chunks por relevância para a pergunta usando o cross-encoder.

        Se não houver chunks (pergunta fora do escopo), retorna lista vazia.
        Se houver menos chunks que top_k, retorna todos reordenados.

        O score do cross-encoder substitui o score de similaridade original
        — valores mais altos indicam maior relevância para a pergunta.
        """
        if not chunks:
            return []

        # Monta pares (pergunta, chunk) para o cross-encoder pontuar
        pares = [(pergunta, chunk.texto) for chunk in chunks]

        # Pontua todos os pares de uma vez (mais eficiente que um a um)
        scores = self.modelo.predict(pares)

        # Associa cada chunk ao seu novo score e reordena
        chunks_com_score = [
            ResultadoBusca(
                doc    = chunk.doc,
                pagina = chunk.pagina,
                texto  = chunk.texto,
                score  = round(float(score), 4),
            )
            for chunk, score in zip(chunks, scores)
        ]

        chunks_reordenados = sorted(
            chunks_com_score,
            key=lambda c: c.score,
            reverse=True,  # maior score = mais relevante
        )
        # Filtra chunks com score negativo — cross-encoder confirmou irrelevância
        chunks_positivos = [c for c in chunks_reordenados if c.score > 0]

        # Se todos foram descartados, retorna lista vazia (fora do escopo)
        top_chunks = chunks_positivos[:top_k]

        logger.info(
            f"Reranking: {len(chunks)} → {len(top_chunks)} chunks "
            f"(scores: {[c.score for c in top_chunks]})"
        )

        return top_chunks