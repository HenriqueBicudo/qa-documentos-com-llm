"""
indexing/embed.py — Geração de embeddings e indexação no ChromaDB

Carrega os chunks gerados pelo chunking.py, gera um embedding para cada um
usando o modelo multilingual-e5-large (roda 100% local), e salva tudo no
ChromaDB persistido em disco.

O modelo e5 exige prefixo nos textos:
    - Chunks de documentos: "passage: <texto>"
    - Perguntas na busca:   "query: <texto>"

Execute:
    python -m indexing.embed
"""

import json
import torch
import chromadb

from pathlib             import Path
from sentence_transformers import SentenceTransformer
from config.logger       import get_logger

logger = get_logger(__name__)

# ── Caminhos ──────────────────────────────────────────────────────────────────

CHUNKS_FILE = Path(__file__).parent / "chunks.json"
CHROMA_DIR  = Path(__file__).parent.parent / "chroma_db"

# ── Configurações ─────────────────────────────────────────────────────────────

MODELO_EMBEDDING = "intfloat/multilingual-e5-large"
COLLECTION_NAME  = "ipardes_rag"
BATCH_SIZE       = 32  # quantos chunks processar por vez (ajuste conforme RAM/VRAM)

# Usa GPU se disponível, senão CPU - Perguntar pro professor se GPU é permitida e se usamos CUDA (NVIDIA) ou ROCm (AMD)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gerar_embeddings(chunks: list[dict], modelo: SentenceTransformer) -> list[list[float]]:
    """
    Gera embeddings para uma lista de chunks em batches.

    O prefixo "passage: " é obrigatório para o modelo e5 — ele foi treinado
    com esse prefixo nos documentos e sem ele a qualidade cai bastante.

    Processa em batches pra não estourar a VRAM da GPU.
    """
    textos = [f"passage: {chunk['texto']}" for chunk in chunks]
    embeddings = []

  # Aqui a gente processa os textos em batches pra não estourar a VRAM da GPU — o modelo e5 é grande e pode consumir muita memória dependendo do tamanho dos chunks
  # Sendo assim o for percorre os textos em passos de BATCH_SIZE, criando sublistas (batches) e processando cada uma separadamente.
    for i in range(0, len(textos), BATCH_SIZE):
        batch = textos[i : i + BATCH_SIZE]
        logger.info(f"  Embedding batch {i // BATCH_SIZE + 1}/{(len(textos) + BATCH_SIZE - 1) // BATCH_SIZE}")

        batch_embeddings = modelo.encode(
            batch,
            normalize_embeddings=True,  # normalização L2 — necessária pra busca por cosseno
            show_progress_bar=False,
        )
        embeddings.extend(batch_embeddings.tolist())

    return embeddings


def indexar_no_chroma(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """
    Salva os chunks e seus embeddings no ChromaDB persistido em disco.

    Estrutura salva por chunk:
        - id:        identificador único (doc_pPAGINA_cINDICE)
        - embedding: vetor gerado pelo e5
        - document:  texto do chunk (usado no retorno da busca)
        - metadata:  doc e pagina (usados nas citações)
    """
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # busca por similaridade de cosseno
    )

    # Insere em batches pra evitar timeout no ChromaDB
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks     = chunks[i : i + BATCH_SIZE]
        batch_embeddings = embeddings[i : i + BATCH_SIZE]
      # Aqui a gente insere os dados no ChromaDB em batches — o método upsert é usado pra inserir ou atualizar os dados, e recebe listas de ids, embeddings, documentos e metadados. 
      # O id é construído com base no nome do documento, número da página e índice do chunk pra garantir unicidade.
        collection.upsert(
            ids        = [c["id"] for c in batch_chunks],
            embeddings = batch_embeddings,
            documents  = [c["texto"] for c in batch_chunks],
            metadatas  = [{"doc": c["doc"], "pagina": c["pagina"]} for c in batch_chunks],
        )

    logger.info(f"Total indexado: {collection.count()} chunks no ChromaDB")


def main():
    # Carrega chunks
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    logger.info(f"Chunks carregados: {len(chunks)}")

    # Carrega modelo na GPU (ou CPU)
    logger.info(f"Carregando modelo {MODELO_EMBEDDING} em {DEVICE}...")
    modelo = SentenceTransformer(MODELO_EMBEDDING, device=DEVICE)
    logger.info("Modelo carregado.")

    # Gera embeddings
    logger.info("Gerando embeddings...")
    embeddings = gerar_embeddings(chunks, modelo)
    logger.info(f"Embeddings gerados: {len(embeddings)}")

    # Indexa no ChromaDB
    logger.info("Indexando no ChromaDB...")
    indexar_no_chroma(chunks, embeddings)
    logger.info("Indexação concluída!")


if __name__ == "__main__":
    main()