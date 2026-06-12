"""
indexing/embed_tabelas.py — Indexação das tabelas extraídas no ChromaDB

As tabelas em tabelas_extraidas.json são convertidas para texto em formato
markdown e inseridas na mesma coleção ChromaDB do texto corrido, com
metadata tipo="tabela". Usa upsert — seguro rodar mais de uma vez.

IDs seguem o padrão doc_pPAGINA_tINDICE para não colidir com os chunks
de texto (que usam _cINDICE).

─── Por que este script foi criado separado do embed.py ──────────────────────

Na versão inicial, o pipeline de ingestão (ingestor.py) já extraía tabelas e
o extract.py já as salvava em tabelas_extraidas.json. Porém o embed.py
carregava apenas chunks.json (texto corrido) e as tabelas ficavam geradas mas
nunca indexadas — dados estruturados dos PDFs eram invisíveis para o RAG.

O objetivo deste script é corrigir essa lacuna: indexar as tabelas na mesma
coleção ChromaDB sem precisar re-embedar todo o texto. Como usa upsert, pode
ser rodado de forma incremental sobre um ChromaDB já existente.

Execute após embed.py já ter indexado o texto:
    python -m indexing.embed_tabelas
"""

import json
import re
import torch
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer
from config.logger import get_logger

logger = get_logger(__name__)

# ── Caminhos ──────────────────────────────────────────────────────────────────

TABELAS_FILE    = Path(__file__).parent / "tabelas_extraidas.json"
CHROMA_DIR      = Path(__file__).parent.parent / "chroma_db"

# ── Configurações ─────────────────────────────────────────────────────────────

# Mesmo modelo do embed.py — vetores precisam estar no mesmo espaço semântico
MODELO_EMBEDDING = "intfloat/multilingual-e5-large"
COLLECTION_NAME  = "ipardes_rag"
BATCH_SIZE       = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Filtro de qualidade ───────────────────────────────────────────────────────

def tabela_tem_conteudo(md: str, min_densidade: float = 0.15) -> bool:
    """
    Retorna True se a tabela tem densidade de conteúdo suficiente para indexar.

    Tabelas com muitas células vazias (| | | | |) produzem embeddings degenerados:
    o vetor resultante acaba próximo da "média" do espaço semântico e tem
    alta similaridade coseno com praticamente qualquer query, incluindo perguntas
    completamente fora do escopo (ex.: "bolo de chocolate" → score 0.89).

    Solução: calcular a fração de chars que não são espaço nem pipe.
    Se < 15%, a tabela é essencialmente estrutura vazia e não deve ser indexada.
    """
    content_chars = len(re.sub(r'[\s|]', '', md))
    return content_chars / max(len(md), 1) >= min_densidade


# ── Preparação dos chunks de tabela ──────────────────────────────────────────

def preparar_chunks_tabelas(tabelas: list[dict]) -> list[dict]:
    """
    Converte cada entrada de tabelas_extraidas.json num chunk indexável.

    Com Docling, o campo "tabela" já é uma string markdown — sem conversão.

    Histórico:
        Versão 1 (pdfplumber + camelot): "tabela" era uma lista de dicts
            [{"col1": "val1", ...}, ...]. Era necessário converter para
            markdown aqui via tabela_para_markdown() antes de embedar.
        Versão atual (Docling): "tabela" já chega como string markdown
            do export_to_markdown() do Docling — conversão eliminada.
    """
    chunks: list[dict] = []
    contagem: dict[str, int] = {}

    for item in tabelas:
        doc    = item["doc"]
        pagina = item["pagina"]
        chave  = f"{doc}_p{pagina}"

        texto = item["tabela"]  # já é markdown — direto do Docling
        if not texto.strip():
            continue
        if not tabela_tem_conteudo(texto):
            logger.debug(f"Tabela descartada (baixa densidade): {doc} p.{pagina}")
            continue

        idx = contagem.get(chave, 0)
        contagem[chave] = idx + 1

        chunks.append({
            "id"    : f"{chave}_t{idx}",
            "doc"   : doc,
            "pagina": pagina,
            "tipo"  : "tabela",
            "texto" : texto,
        })

    return chunks


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    with open(TABELAS_FILE, encoding="utf-8") as f:
        tabelas = json.load(f)
    logger.info(f"Tabelas carregadas: {len(tabelas)}")

    chunks = preparar_chunks_tabelas(tabelas)
    logger.info(f"Chunks de tabela preparados: {len(chunks)}")

    if not chunks:
        logger.warning("Nenhuma tabela válida para indexar.")
        return

    logger.info(f"Carregando modelo {MODELO_EMBEDDING} em {DEVICE}...")
    modelo = SentenceTransformer(MODELO_EMBEDDING, device=DEVICE)
    logger.info("Modelo carregado.")

    # Gera embeddings com o mesmo prefixo "passage: " do embed.py
    textos = [f"passage: {c['texto']}" for c in chunks]
    embeddings: list[list[float]] = []

    for i in range(0, len(textos), BATCH_SIZE):
        batch = textos[i : i + BATCH_SIZE]
        logger.info(f"  Embedding batch {i // BATCH_SIZE + 1}/{(len(textos) + BATCH_SIZE - 1) // BATCH_SIZE}")
        batch_emb = modelo.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name     = COLLECTION_NAME,
        metadata = {"hnsw:space": "cosine"},
    )

    # Remove todos os chunks de tabela existentes antes de reinserir.
    # Necessário para que tabelas filtradas pelo critério de qualidade
    # (adicionado posteriormente) sejam também removidas do ChromaDB.
    # O upsert sozinho não remove IDs que deixaram de ser gerados.
    existentes = collection.get(where={"tipo": "tabela"})
    if existentes["ids"]:
        collection.delete(ids=existentes["ids"])
        logger.info(f"Removidos {len(existentes['ids'])} chunks de tabela antigos")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks     = chunks[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]
        collection.upsert(
            ids        = [c["id"]    for c in batch_chunks],
            embeddings = batch_embeddings,
            documents  = [c["texto"] for c in batch_chunks],
            metadatas  = [
                {"doc": c["doc"], "pagina": c["pagina"], "tipo": c["tipo"]}
                for c in batch_chunks
            ],
        )

    logger.info(f"Total no ChromaDB após tabelas: {collection.count()} chunks")


if __name__ == "__main__":
    main()
