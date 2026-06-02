"""
tests/test_chunking.py — Testes unitários do pipeline de chunking

Verifica:
    - Qualidade do chunking (tamanho, overlap, IDs únicos)
    - Integridade dos metadados (doc, pagina presentes em todos os chunks)
    - Consistência dos arquivos intermediários gerados
    - Comportamento com entradas extremas (texto vazio, muito curto, muito longo)
"""

import json
import pytest
from pathlib import Path
from indexing.chunking import chunkar_paginas, CHUNK_SIZE, CHUNK_OVERLAP

CHUNKS_FILE  = Path(__file__).parent.parent / "indexing" / "chunks.json"
PAGINAS_FILE = Path(__file__).parent.parent / "indexing" / "paginas_extraidas.json"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def chunks_reais():
    assert CHUNKS_FILE.exists(), "chunks.json não encontrado — rode indexing/chunking.py primeiro"
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def paginas_reais():
    assert PAGINAS_FILE.exists(), "paginas_extraidas.json não encontrado — rode indexing/extract.py primeiro"
    with open(PAGINAS_FILE, encoding="utf-8") as f:
        return json.load(f)


# ── Testes de estrutura dos chunks reais ──────────────────────────────────────

def test_chunks_nao_vazios(chunks_reais):
    """O arquivo de chunks deve ter pelo menos um chunk."""
    assert len(chunks_reais) > 0


def test_chunks_quantidade_minima(chunks_reais):
    """Com 3 PDFs extensos, esperamos pelo menos 200 chunks."""
    assert len(chunks_reais) >= 200, f"Poucos chunks gerados: {len(chunks_reais)}"


def test_chunks_tem_campos_obrigatorios(chunks_reais):
    """Todos os chunks devem ter id, doc, pagina e texto."""
    campos = {"id", "doc", "pagina", "texto"}
    for chunk in chunks_reais:
        assert campos.issubset(chunk.keys()), f"Chunk sem campos obrigatórios: {chunk}"


def test_ids_unicos(chunks_reais):
    """Nenhum chunk pode ter o mesmo ID — unicidade essencial pro ChromaDB."""
    ids = [c["id"] for c in chunks_reais]
    assert len(ids) == len(set(ids)), "IDs duplicados encontrados nos chunks"


def test_ids_seguem_padrao(chunks_reais):
    """IDs devem seguir o padrão doc_pPAGINA_cINDICE."""
    for chunk in chunks_reais:
        partes = chunk["id"].rsplit("_p", 1)
        assert len(partes) == 2, f"ID fora do padrão: {chunk['id']}"
        pagina_indice = partes[1].split("_c")
        assert len(pagina_indice) == 2, f"ID fora do padrão: {chunk['id']}"
        assert pagina_indice[0].isdigit(), f"Página não numérica no ID: {chunk['id']}"
        assert pagina_indice[1].isdigit(), f"Índice não numérico no ID: {chunk['id']}"


def test_chunks_nao_vazios_texto(chunks_reais):
    """Nenhum chunk pode ter texto vazio."""
    for chunk in chunks_reais:
        assert chunk["texto"].strip(), f"Chunk com texto vazio: {chunk['id']}"


def test_chunks_respeitam_tamanho_maximo(chunks_reais):
    """
    Nenhum chunk deve ultrapassar significativamente o CHUNK_SIZE.
    Tolerância de 10% para casos onde o separador não é encontrado.
    """
    limite = CHUNK_SIZE * 1.1
    violacoes = [c for c in chunks_reais if len(c["texto"]) > limite]
    assert len(violacoes) == 0, (
        f"{len(violacoes)} chunks excedem o limite de {limite:.0f} chars: "
        f"{[v['id'] for v in violacoes[:3]]}"
    )


def test_chunks_tem_tamanho_minimo(chunks_reais):
    """Chunks muito curtos (< 50 chars) indicam problema no chunking ou limpeza."""
    curtos = [c for c in chunks_reais if len(c["texto"]) < 50]
    proporcao = len(curtos) / len(chunks_reais)
    assert proporcao < 0.02, (
        f"{len(curtos)} chunks muito curtos ({proporcao:.1%} do total)"
    )


def test_docs_conhecidos(chunks_reais):
    """Os chunks devem vir apenas dos 3 documentos esperados."""
    docs_esperados = {
        "desenvolvimento_paranaense",
        "analise_conjuntural",
        "avaliacoes_politicas_publicas",
    }
    docs_encontrados = {c["doc"] for c in chunks_reais}
    assert docs_encontrados.issubset(docs_esperados), (
        f"Documentos inesperados: {docs_encontrados - docs_esperados}"
    )


def test_todos_docs_representados(chunks_reais):
    """Os três documentos devem ter chunks — nenhum pode ter sido ignorado."""
    docs = {c["doc"] for c in chunks_reais}
    assert "desenvolvimento_paranaense"    in docs
    assert "analise_conjuntural"           in docs
    assert "avaliacoes_politicas_publicas" in docs


def test_paginas_sao_inteiros(chunks_reais):
    """O campo pagina deve ser um inteiro positivo."""
    for chunk in chunks_reais:
        assert isinstance(chunk["pagina"], int), f"pagina não é int: {chunk['id']}"
        assert chunk["pagina"] > 0, f"pagina <= 0: {chunk['id']}"


def test_texto_e_string(chunks_reais):
    """O campo texto deve ser uma string."""
    for chunk in chunks_reais:
        assert isinstance(chunk["texto"], str), f"texto não é string: {chunk['id']}"


def test_chunks_por_documento(chunks_reais):
    """Cada documento deve ter uma quantidade razoável de chunks."""
    contagem = {}
    for c in chunks_reais:
        contagem[c["doc"]] = contagem.get(c["doc"], 0) + 1

    for doc, qtd in contagem.items():
        assert qtd >= 10, f"Documento {doc} tem poucos chunks: {qtd}"


# ── Testes de overlap ─────────────────────────────────────────────────────────

def test_overlap_entre_chunks_consecutivos(chunks_reais):
    """
    Chunks consecutivos da mesma página devem compartilhar conteúdo (overlap),
    garantindo que o contexto não seja perdido nas bordas dos chunks.
    """
    chunks_por_pagina: dict[str, list] = {}
    for chunk in chunks_reais:
        chave = f"{chunk['doc']}_{chunk['pagina']}"
        chunks_por_pagina.setdefault(chave, []).append(chunk)

    pares_testados = pares_com_overlap = 0

    for grupo in chunks_por_pagina.values():
        if len(grupo) < 2:
            continue
        grupo_ord = sorted(grupo, key=lambda c: int(c["id"].split("_c")[-1]))

        for i in range(len(grupo_ord) - 1):
            t_atual  = grupo_ord[i]["texto"]
            t_proximo = grupo_ord[i + 1]["texto"]
            if len(t_atual) < 100 or len(t_proximo) < 100:
                continue
            pares_testados += 1
            fim_atual = t_atual[-50:].strip()
            if fim_atual and fim_atual in t_proximo:
                pares_com_overlap += 1

    if pares_testados > 0:
        taxa = pares_com_overlap / pares_testados
        assert taxa >= 0.5, (
            f"Overlap insuficiente: {taxa:.1%} dos pares consecutivos "
            f"({pares_com_overlap}/{pares_testados})"
        )


def test_chunk_overlap_config_valida():
    """CHUNK_OVERLAP deve ser menor que CHUNK_SIZE."""
    assert CHUNK_OVERLAP < CHUNK_SIZE, (
        f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) >= CHUNK_SIZE ({CHUNK_SIZE})"
    )


def test_chunk_overlap_proporcao_razoavel():
    """
    CHUNK_OVERLAP deve ser entre 5% e 50% do CHUNK_SIZE.
    Muito pequeno = contexto perdido; muito grande = redundância excessiva.
    """
    proporcao = CHUNK_OVERLAP / CHUNK_SIZE
    assert 0.05 <= proporcao <= 0.50, (
        f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) é {proporcao:.1%} do CHUNK_SIZE — fora do intervalo razoável"
    )


# ── Testes unitários de chunkar_paginas ───────────────────────────────────────

def test_chunkar_paginas_simples():
    """Texto longo deve gerar mais de um chunk."""
    texto = "palavra " * 600
    chunks = chunkar_paginas([{"doc": "teste", "pagina": 1, "texto": texto}])
    assert len(chunks) >= 2


def test_chunkar_paginas_texto_curto():
    """Texto curto (menor que CHUNK_SIZE) deve gerar exatamente um chunk."""
    texto = "Texto curto sobre o Paraná."
    chunks = chunkar_paginas([{"doc": "doc", "pagina": 1, "texto": texto}])
    assert len(chunks) == 1
    assert chunks[0]["texto"] == texto


def test_chunkar_paginas_preserva_metadados():
    """Os metadados de origem devem ser preservados em cada chunk."""
    paginas = [
        {"doc": "doc_a", "pagina": 5,  "texto": "Texto A."},
        {"doc": "doc_b", "pagina": 10, "texto": "Texto B."},
    ]
    chunks = chunkar_paginas(paginas)
    docs   = {c["doc"] for c in chunks}
    pags   = {c["pagina"] for c in chunks}
    assert "doc_a" in docs and "doc_b" in docs
    assert 5 in pags and 10 in pags


def test_chunkar_paginas_ids_unicos():
    """IDs gerados devem ser únicos mesmo com múltiplas páginas."""
    texto = "palavra " * 600
    paginas = [
        {"doc": "doc_x", "pagina": i, "texto": texto}
        for i in range(1, 6)
    ]
    chunks = chunkar_paginas(paginas)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunkar_paginas_lista_vazia():
    """Lista vazia deve retornar lista vazia sem erros."""
    assert chunkar_paginas([]) == []


def test_chunkar_paginas_multiplos_docs():
    """chunkar_paginas deve processar múltiplos documentos corretamente."""
    docs = ["doc_a", "doc_b", "doc_c"]
    paginas = [{"doc": d, "pagina": 1, "texto": "conteúdo " * 10} for d in docs]
    chunks = chunkar_paginas(paginas)
    docs_gerados = {c["doc"] for c in chunks}
    assert docs_gerados == set(docs)


def test_chunkar_paginas_indice_sequencial():
    """
    Chunks da mesma página devem ter índices sequenciais começando em 0.
    """
    texto = "palavra " * 600
    chunks = chunkar_paginas([{"doc": "doc", "pagina": 1, "texto": texto}])
    indices = [int(c["id"].split("_c")[-1]) for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunkar_paginas_texto_apenas_espacos():
    """Texto com apenas espaços pode gerar chunks ou nenhum — não deve lançar erro."""
    paginas = [{"doc": "doc", "pagina": 1, "texto": "   "}]
    try:
        chunks = chunkar_paginas(paginas)
        # Se gerou algum chunk, o texto deve ter conteúdo
        for c in chunks:
            assert c["texto"].strip() != ""
    except Exception as e:
        pytest.fail(f"chunkar_paginas lançou exceção inesperada: {e}")


def test_chunkar_preserva_conteudo_semantico():
    """
    O conteúdo essencial do texto deve aparecer em pelo menos um chunk.
    """
    conteudo_chave = "IPARDES desenvolvimento econômico Paraná"
    texto = f"Introdução. {conteudo_chave}. " + "texto de preenchimento " * 200
    chunks = chunkar_paginas([{"doc": "doc", "pagina": 1, "texto": texto}])
    textos_unidos = " ".join(c["texto"] for c in chunks)
    assert "IPARDES" in textos_unidos
    assert "Paraná"  in textos_unidos
