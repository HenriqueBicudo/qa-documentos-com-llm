"""
ingestion/ingestor.py — Carregamento e estruturação dos documentos PDF

O PDFLoader é a peça central da ingestão. Para cada página de um PDF ele:
    1. Extrai as tabelas com TableExtractor (pdfplumber + camelot)
    2. Filtra o texto removendo as áreas onde ficam as tabelas — assim
       texto e tabelas ficam separados e limpos, sem duplicação
    3. Retorna um RawDocument por página, com texto e tabelas distintos

Isso é importante pro RAG: o texto corrido vai pro chunker, e as tabelas
ficam indexadas separadamente com seus dados estruturados.
"""

from ingestion.table_extractor import TableExtractor

from typing      import List
from pathlib     import Path
from dataclasses import dataclass, field

from config.logger   import get_logger
from pdfplumber.page import Page

logger = get_logger(__name__)

# Alias: uma tabela bruta é uma lista de linhas, cada linha é uma lista de células
TableData = List[List[str | None]]


@dataclass
class RawDocument:
    """
    Representa o conteúdo extraído de uma única página de um PDF.

    Campos:
        page:        número da página (começa em 1)
        total_pages: total de páginas do documento
        source:      caminho do arquivo PDF de origem
        content:     texto corrido da página (sem as áreas de tabela)
        tables:      lista de tabelas extraídas da página (dados brutos)
        metadata:    dicionário livre para metadados extras (ex: nome do doc)
    """
    page       : int
    total_pages: int
    source     : str
    content    : str
    tables     : List[TableData] = field(default_factory=list)
    metadata   : dict            = field(default_factory=dict)


class PDFLoader:
    """
    Carrega um PDF e retorna uma lista de RawDocuments (um por página).

    Uso:
        loader = PDFLoader("arquivo.pdf")
        documentos = loader.load()
    """

    def __init__(self, pdf_path: str | Path):
        self.path = Path(pdf_path)

    def _group_tables_by_page(
        self,
        extractor: TableExtractor,
    ) -> dict[int, list[TableData]]:
        """
        Extrai todas as tabelas do PDF e as organiza por número de página.

        Retorna um dict: {numero_pagina: [tabela1, tabela2, ...]}
        """
        tables_per_page: dict[int, list[TableData]] = {}

        for table in extractor.extract_all_tables():
            if table.page is None:
                continue

            tables_per_page.setdefault(table.page, []).append(table.data)

        return tables_per_page

    def _build_document(
        self,
        page           : Page,
        total_pages    : int,
        tables_per_page: dict[int, list[TableData]],
        extractor      : TableExtractor,
    ) -> RawDocument | None:
        """
        Constrói um RawDocument para uma página.

        O ponto chave aqui é o filtro de texto: antes de extrair o texto
        corrido, a função calcula as bounding boxes das tabelas na página
        e filtra os objetos de texto que estão dentro dessas áreas.

        Isso garante que o campo `content` não contenha texto de tabelas,
        evitando duplicação quando ambos forem indexados no RAG.

        Retorna None se a página não tiver nem texto nem tabelas.
        """
        table_areas = extractor._build_table_areas(page.page_number)
        tables      = tables_per_page.get(page.page_number, [])
        height      = page.height

        # Converte as áreas do formato camelot (y de baixo pra cima)
        # de volta pro formato pdfplumber (top de cima pra baixo)
        if table_areas:
            bboxes = []
            for area in table_areas:
                x1, y1, x2, y2 = map(float, area.split(","))
                bboxes.append((x1, height - y2, x2, height - y1))
        else:
            # Fallback: usa as bboxes detectadas automaticamente pelo pdfplumber
            bboxes = [table.bbox for table in page.find_tables()]

        # Filtra os objetos de texto que estão dentro das áreas de tabela
        filtered_page = page.filter(
            lambda obj: not any(
                obj["x0"]    >= bbox[0] and obj["x1"]     <= bbox[2] and
                obj["top"]   >= bbox[1] and obj["bottom"] <= bbox[3]
                for bbox in bboxes
            )
        )

        text = (filtered_page.extract_text() or "").strip()

        # Descarta páginas completamente vazias
        if not text and not tables:
            return None

        return RawDocument(
            page        = page.page_number,
            total_pages = total_pages,
            source      = str(self.path),
            content     = text,
            tables      = tables,
        )

    def load(self) -> list[RawDocument]:
        """
        Abre o PDF, extrai texto e tabelas por página e retorna
        a lista de RawDocuments prontos para o pipeline de chunking.
        """
        logger.info(f"Carregando PDF: {self.path.name}")
        documents: list[RawDocument] = []

        with TableExtractor(self.path) as extractor:
            tables_per_page = self._group_tables_by_page(extractor)
            total_pages     = len(extractor.pdf.pages)

            for page in extractor.pdf.pages:
                document = self._build_document(
                    page,
                    total_pages,
                    tables_per_page,
                    extractor,
                )

                if document:
                    documents.append(document)

        logger.info(f"  → {len(documents)} páginas carregadas")
        return documents
