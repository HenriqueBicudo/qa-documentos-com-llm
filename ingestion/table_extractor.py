"""
ingestion/table_extractor.py — Localização e extração de tabelas dos PDFs

Combina duas ferramentas:
    - pdfplumber: localiza onde as tabelas estão na página, usando os
      marcadores textuais "TABELA N" (início) e "FONTE" (fim).
    - camelot: extrai o conteúdo das tabelas com precisão, usando as
      coordenadas calculadas pelo pdfplumber.

Nota sobre sistemas de coordenadas:
    pdfplumber mede `top` de cima pra baixo (0 = topo da página).
    camelot    mede `y`   de baixo pra cima (0 = base da página).
    Conversão: y_camelot = altura_pagina - top_pdfplumber
"""

from typing      import Generator
from pathlib     import Path
from dataclasses import dataclass

import re
import pdfplumber
import camelot.io as camelot
from   camelot.core import Table, TableList

# ── Padrões regex ─────────────────────────────────────────────────────────────

# Detecta "TABELA 1", "TABELA 2", etc. — início de uma tabela
TABLE_PATTERN = re.compile(r"\bTABELA\s+\d+\b")

# Detecta linhas que começam com "FONTE" — fim de uma tabela
SOURCE_PATTERN = re.compile(r"^\s*FONTE\b")

# ── Tipos auxiliares ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TextLine:
    """Linha de texto com sua posição vertical na página (em pontos)."""
    text: str
    top : float


# ── Funções auxiliares ────────────────────────────────────────────────────────

def group_into_lines(
    words: list[dict],
    tolerance_y: int = 3
) -> list[list[dict]]:
    """
    Agrupa palavras soltas em linhas lógicas de texto.

    O pdfplumber extrai palavras individualmente, cada uma com sua
    posição (x0, top). Palavras na mesma linha têm valores de `top`
    muito próximos — se a diferença for <= tolerance_y (padrão: 3px),
    elas pertencem à mesma linha.
    """
    if not words:
        return []

    lines: list[list[dict]] = []
    current_line = [words[0]]

    for word in words[1:]:
        same_line = abs(word["top"] - current_line[-1]["top"]) <= tolerance_y

        if same_line:
            current_line.append(word)
        else:
            lines.append(sorted(current_line, key=lambda w: w["x0"]))
            current_line = [word]

    # Adiciona a última linha que ficou fora do loop
    lines.append(sorted(current_line, key=lambda w: w["x0"]))
    return lines


def format_area(x1: float, y1: float, x2: float, y2: float) -> str:
    """
    Formata coordenadas no padrão do camelot: "x1,y1,x2,y2".

    O camelot espera:
        (x1, y1) = canto inferior esquerdo
        (x2, y2) = canto superior direito
    (lembrando que y cresce de baixo pra cima no camelot)
    """
    return f"{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}"


def line_text(words: list[dict]) -> str:
    """Junta uma lista de palavras em uma única string."""
    return " ".join(word["text"] for word in words)


# ── Classe principal ──────────────────────────────────────────────────────────

class TableExtractor:
    """
    Extrai tabelas de um PDF usando pdfplumber (para localizar)
    e camelot (para extrair o conteúdo).

    Uso básico:
        with TableExtractor("arquivo.pdf") as extractor:
            tables = extractor.extract_all_tables()
    """

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = str(pdf_path)
        self.pdf      = pdfplumber.open(self.pdf_path)

    def close(self) -> None:
        self.pdf.close()

    def __enter__(self) -> "TableExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        """Garante que o documento é fechado ao sair do bloco `with`."""
        self.close()

    # ── Métodos internos ───────────────────────────────────────────────────────

    def _validate_page(self, page_num: int) -> None:
        """Lança ValueError se page_num estiver fora do intervalo válido."""
        total = len(self.pdf.pages)
        if not (1 <= page_num <= total):
            raise ValueError(
                f"Página {page_num} inválida "
                f"(documento tem {total} páginas)"
            )

    def _get_page_lines(self, page_num: int) -> tuple[TextLine, ...]:
        """
        Extrai todas as linhas de texto de uma página como objetos TextLine.

        Fluxo:
            1. Extrai palavras brutas com pdfplumber
            2. Agrupa em linhas lógicas com group_into_lines
            3. Converte cada grupo em TextLine(text, top)
        """
        self._validate_page(page_num)
        page  = self.pdf.pages[page_num - 1]
        words = page.extract_words(keep_blank_chars=True)

        return tuple(
            TextLine(text=line_text(line), top=line[0]["top"])
            for line in group_into_lines(words)
        )

    def _find_table_pages(self) -> Generator[int, None, None]:
        """
        Gera os números das páginas que contêm tabelas detectadas pelo pdfplumber.
        Usa generator pra não carregar todas as páginas em memória de uma vez.
        """
        return (
            page_num
            for page_num, page in enumerate(self.pdf.pages, start=1)
            if page.find_tables()
        )

    def _build_table_areas(self, page_num: int) -> list[str]:
        """
        Localiza onde cada tabela começa e termina na página e retorna
        suas coordenadas no formato que o camelot espera.

        Estratégia:
            - Início da tabela: linha com "TABELA {N}"
            - Fim da tabela:    próxima linha que começa com "FONTE"
                                (fallback: final da página)
        """
        page          = self.pdf.pages[page_num - 1]
        lines         = self._get_page_lines(page_num)
        width, height = page.width, page.height
        areas: list[str] = []

        for index, line in enumerate(lines):
            if not TABLE_PATTERN.search(line.text):
                continue

            y_end = next(
                (
                    following.top
                    for following in lines[index + 1:]
                    if SOURCE_PATTERN.match(following.text)
                ),
                height,  # fallback: vai até o final da página
            )

            areas.append(
                format_area(
                    x1=0,
                    x2=width,
                    y1=height - y_end,
                    y2=height - line.top,
                )
            )

        return areas

    # ── Métodos de extração ────────────────────────────────────────────────────

    def _extract_base(self, page_num: int) -> TableList:
        """
        Extrai tabelas sem áreas específicas (modo lattice).
        O camelot detecta automaticamente — funciona bem com bordas visíveis.
        """
        return camelot.read_pdf(
            self.pdf_path,
            pages=str(page_num),
            flavor="lattice",
        )

    def _extract_anchored(self, page_num: int, table_areas: list[str]) -> TableList:
        """
        Extrai tabelas usando áreas pré-calculadas (modo stream).
        Mais preciso que _extract_base quando temos as coordenadas exatas.
        """
        return camelot.read_pdf(
            self.pdf_path,
            pages=str(page_num),
            flavor="stream",
            table_areas=table_areas,
        )

    def extract_tables_by_page(self, page_num: int) -> TableList:
        """
        Extrai as tabelas de uma página específica.

        Se encontrar marcadores "TABELA N" → usa coordenadas (anchored).
        Caso contrário → deixa o camelot detectar automaticamente (base).
        """
        table_areas = self._build_table_areas(page_num)

        if table_areas:
            return self._extract_anchored(page_num, table_areas)

        return self._extract_base(page_num)

    def extract_all_tables(self) -> list[Table]:
        """
        Extrai todas as tabelas do documento, percorrendo apenas as
        páginas que contêm tabelas detectadas pelo pdfplumber.
        """
        tables: list[Table] = []

        for page_num in self._find_table_pages():
            tables.extend(self.extract_tables_by_page(page_num))

        return tables
