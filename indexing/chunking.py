"""
indexing/chunking.py — Divisão do texto extraído em chunks para indexação

O chunking é a etapa que divide as páginas em pedaços menores prontos
para serem vetorizados. Usamos o RecursiveCharacterTextSplitter do LangChain
que tenta dividir o texto respeitando a estrutura natural do documento:
primeiro tenta quebrar em parágrafos, depois em frases, depois em palavras.

Parâmetros escolhidos:
    chunk_size    = 512  tokens — tamanho máximo de cada chunk
    chunk_overlap = 64   tokens — sobreposição entre chunks consecutivos

Execute:
    python -m indexing.chunking
"""

import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.logger import get_logger

logger = get_logger(__name__)

# ── Caminhos ──────────────────────────────────────────────────────────────────

INPUT_FILE  = Path(__file__).parent / "paginas_extraidas.json"
OUTPUT_FILE = Path(__file__).parent / "chunks.json"

# ── Parâmetros de chunking ────────────────────────────────────────────────────

# chunk_size em caracteres — aproximadamente 512 tokens para português assim encaixando no contexto do modelo (4096 tokens)
# (1 token ≈ 4 caracteres em média para textos em PT)
CHUNK_SIZE    = 2048
CHUNK_OVERLAP = 256

def chunkar_paginas(paginas: list[dict]) -> list[dict]:
    """
    Divide cada página em chunks menores usando RecursiveCharacterTextSplitter.

    O splitter tenta quebrar o texto respeitando a estrutura natural:
        1. Primeiro tenta quebrar em parágrafos (\n\n)
        2. Depois em frases (\n)
        3. Depois em palavras (espaço)
        4. Por último em caracteres (fallback)

    Cada chunk herda os metadados da página de origem:
        doc, pagina — essenciais para as citações nas respostas do RAG
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []

    for pagina in paginas:
        textos_divididos = splitter.split_text(pagina["texto"])

        for i, texto in enumerate(textos_divididos):
            chunks.append({
                "id"    : f"{pagina['doc']}_p{pagina['pagina']}_c{i}",
                "doc"   : pagina["doc"],
                "pagina": pagina["pagina"],
                "texto" : texto,
            })

    return chunks
  
def main():
    # Carrega as páginas extraídas
    with open(INPUT_FILE, encoding="utf-8") as f:
        paginas = json.load(f)

    logger.info(f"Páginas carregadas: {len(paginas)}")

    chunks = chunkar_paginas(paginas)

    logger.info(f"Chunks gerados: {len(chunks)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info(f"Chunks salvos em {OUTPUT_FILE}")


if __name__ == "__main__":
    main()