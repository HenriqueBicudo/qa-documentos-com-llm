"""
indexing/extract.py — Script principal de extração e limpeza dos PDFs

Orquestra o pipeline de ingestão:
    1. Carrega cada PDF com PDFLoader (texto + tabelas separados)
    2. Aplica filtros de qualidade (páginas muito curtas, sumários)
    3. Salva o resultado em dois arquivos JSON:
        - paginas_extraidas.json : texto corrido por página
        - tabelas_extraidas.json : tabelas estruturadas por página

Execute:
    python indexing/extract.py
"""

import json
import re
from pathlib import Path

from ingestion.ingestor import PDFLoader, RawDocument
from config.logger      import get_logger

logger = get_logger(__name__)

# ── Caminhos ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent  # salva em indexing/

PDFS = {
    "desenvolvimento_paranaense"   : DATA_DIR / "desenvolvimento_paranaense.pdf",
    "analise_conjuntural"          : DATA_DIR / "Analise_Conjuntural_julho_agosto_2025.pdf",
    "avaliacoes_politicas_publicas": DATA_DIR / "Avaliacoes Politicas Publicas Brasil_revisao escopo.pdf",
}

# ── Filtros de qualidade ──────────────────────────────────────────────────────

# Títulos de cabeçalho repetidos em toda página — removidos linha por linha
TITULOS_CABECALHO = [
    "DESENVOLVIMENTO PARANAENSE: CONTEXTO, TENDÊNCIAS E DESAFIOS",
    "ANÁLISE CONJUNTURAL",
    "AVALIAÇÕES DE POLÍTICAS PÚBLICAS",
]


def pagina_e_valida(texto: str) -> bool:
    """
    Descarta páginas sem conteúdo útil pro RAG.

    Critérios de descarte:
        - Texto muito curto (< 150 chars): capas, separadores, páginas em branco
        - Muitos '....': sumários e índices com pontilhado de página
    """
    texto_limpo = texto.strip()

    if len(texto_limpo) < 150:
        return False

    if texto_limpo.count("....") > 5:
        return False
  
    # Ficha técnica: muitas linhas com " - " (cargo - nome)
    linhas_com_traco = sum(1 for l in texto_limpo.split("\n") if " - " in l)
    if linhas_com_traco > 6:
        return False

    # Referências bibliográficas: começa com "REFERÊNCIAS"
    if texto_limpo.upper().startswith("REFERÊNCIAS"):
        return False

    return True
  
  
def limpar_texto(texto: str) -> str:
    """
    Remove ruídos comuns do texto extraído:
        - Números de página soltos (linhas que são só um dígito)
        - Títulos de documento repetidos em cada página como cabeçalho
        - Espaços e quebras de linha extras
    """
    linhas = texto.split("\n")
    linhas_limpas = []

    for linha in linhas:
        linha_strip = linha.strip()

        # Remove linhas que são só um número (número de página)
        if linha_strip.isdigit():
            continue

        # Remove linhas que são o título do documento repetido como cabeçalho
        if any(titulo in linha_strip.upper() for titulo in TITULOS_CABECALHO):
            continue

        linhas_limpas.append(linha)

    # Junta e normaliza espaços múltiplos
    texto_limpo = "\n".join(linhas_limpas)
    texto_limpo = " ".join(texto_limpo.split())

    return texto_limpo.strip()


# ── Pipeline principal ────────────────────────────────────────────────────────

def extrair_todos(pdfs: dict[str, Path]) -> tuple[list[dict], list[dict]]:
    """
    Executa o pipeline de extração em todos os PDFs.

    Retorna dois lists:
        - paginas: [{doc, pagina, texto}, ...]
        - tabelas: [{doc, pagina, tabela}, ...]
    """
    todas_paginas: list[dict] = []
    todas_tabelas: list[dict] = []

    for nome_doc, caminho in pdfs.items():
        if not caminho.exists():
            logger.warning(f"PDF não encontrado: {caminho.name}")
            continue

        logger.info(f"Processando: {nome_doc}")
        loader    = PDFLoader(caminho)
        documentos = loader.load()

        paginas_validas = 0

        for doc in documentos:
            # ── Texto corrido ──────────────────────────────────────────────
            if doc.content and pagina_e_valida(doc.content):
                texto_limpo = limpar_texto(doc.content)
                if texto_limpo:
                    todas_paginas.append({
                        "doc"   : nome_doc,
                        "pagina": doc.page,
                        "texto" : texto_limpo,
                    })
                    paginas_validas += 1

            # ── Tabelas estruturadas ───────────────────────────────────────
            # Com Docling, doc.tables já são strings markdown — sem conversão.
            # Antes (pdfplumber + camelot): era necessário parsear manualmente
            # cabeçalhos e linhas de listas de listas, convertendo para dicts.
            for tabela_md in doc.tables:
                if tabela_md.strip():
                    todas_tabelas.append({
                        "doc"   : nome_doc,
                        "pagina": doc.page,
                        "tabela": tabela_md,
                    })

        logger.info(f"  → {paginas_validas} páginas válidas | {len(todas_tabelas)} tabelas acumuladas")

    return todas_paginas, todas_tabelas


def main():
    paginas, tabelas = extrair_todos(PDFS)

    # Salva texto corrido
    out_paginas = OUT_DIR / "paginas_extraidas.json"
    with open(out_paginas, "w", encoding="utf-8") as f:
        json.dump(paginas, f, ensure_ascii=False, indent=2)
    logger.info(f"Texto salvo: {len(paginas)} páginas → {out_paginas}")

    # Salva tabelas
    out_tabelas = OUT_DIR / "tabelas_extraidas.json"
    with open(out_tabelas, "w", encoding="utf-8") as f:
        json.dump(tabelas, f, ensure_ascii=False, indent=2)
    logger.info(f"Tabelas salvas: {len(tabelas)} tabelas → {out_tabelas}")


if __name__ == "__main__":
    main()
