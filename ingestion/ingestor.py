"""
ingestion/ingestor.py — Carregamento e estruturação dos documentos PDF via Docling

─── Histórico de versões ─────────────────────────────────────────────────────

Versão 1 — pdfplumber + camelot:
    Extração manual combinando dois sistemas de coordenadas diferentes
    (pdfplumber mede de cima pra baixo, camelot de baixo pra cima).
    Requeria Ghostscript instalado no sistema operacional.
    Localizava tabelas por marcadores textuais "TABELA N" / "FONTE".
    Código espalhado em ~350 linhas entre ingestor.py e table_extractor.py.

Versão atual — Docling (IBM, 2024):
    A biblioteca usa modelos ML internos para entender o layout do documento,
    separar texto de tabelas e exportar tudo de forma estruturada.
    Sem Ghostscript. Tabelas chegam prontas em markdown. Código em ~60 linhas.
    Instalação: pip install docling

─── Funcionamento ────────────────────────────────────────────────────────────

Para cada PDF o Docling:
    1. Analisa o layout de cada página com modelos de visão computacional
    2. Separa blocos de texto de tabelas automaticamente
    3. Exporta cada tabela em markdown (pronto para embedding)

Retorna um RawDocument por página com:
    - content: texto corrido (sem as tabelas)
    - tables:  lista de strings markdown, uma por tabela da página
"""

from dataclasses import dataclass, field
from pathlib     import Path
from typing      import List
from collections import defaultdict

from docling.datamodel.base_models      import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter         import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend  import PyPdfiumDocumentBackend
from config.logger                      import get_logger

logger = get_logger(__name__)


@dataclass
class RawDocument:
    """
    Representa o conteúdo extraído de uma única página de um PDF.

    Campos:
        page:        número da página (começa em 1)
        total_pages: total de páginas do documento
        source:      caminho do arquivo PDF de origem
        content:     texto corrido da página (sem tabelas)
        tables:      lista de tabelas em formato markdown
        metadata:    dicionário livre para metadados extras
    """
    page       : int
    total_pages: int
    source     : str
    content    : str
    tables     : List[str] = field(default_factory=list)
    metadata   : dict      = field(default_factory=dict)


class PDFLoader:
    """
    Carrega um PDF via Docling e retorna uma lista de RawDocuments (um por página).

    Uso:
        loader = PDFLoader("arquivo.pdf")
        documentos = loader.load()
    """

    def __init__(self, pdf_path: str | Path):
        self.path = Path(pdf_path)

        # Backend PyPdfium2: extrai texto direto da estrutura do PDF sem
        # renderizar páginas como imagens — evita o std::bad_alloc que o
        # backend padrão (docling-parse) causava ao processar documentos
        # grandes (falhava a partir da página 17, perdendo ~80% do conteúdo).
        pipeline_options = PdfPipelineOptions(do_ocr=False)
        self._converter  = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options = pipeline_options,
                    backend          = PyPdfiumDocumentBackend,
                )
            }
        )

    def load(self) -> list[RawDocument]:
        """
        Converte o PDF e agrupa texto e tabelas por página.
        """
        logger.info(f"Carregando PDF: {self.path.name}")

        doc = self._converter.convert(str(self.path)).document

        # Agrupamento por página
        pages_text  : dict[int, list[str]] = defaultdict(list)
        pages_tables: dict[int, list[str]] = defaultdict(list)

        for item in doc.texts:
            if item.prov and item.text.strip():
                pages_text[item.prov[0].page_no].append(item.text)

        for table in doc.tables:
            if table.prov:
                md = table.export_to_markdown(doc)
                if md.strip():
                    pages_tables[table.prov[0].page_no].append(md)

        all_pages   = set(pages_text.keys()) | set(pages_tables.keys())
        total_pages = max(all_pages) if all_pages else 0

        documents: list[RawDocument] = []
        for page_no in sorted(all_pages):
            content = "\n".join(pages_text.get(page_no, []))
            tables  = pages_tables.get(page_no, [])

            if not content.strip() and not tables:
                continue

            documents.append(RawDocument(
                page        = page_no,
                total_pages = total_pages,
                source      = str(self.path),
                content     = content,
                tables      = tables,
            ))

        logger.info(f"  → {len(documents)} páginas carregadas")
        return documents
