"""
tests/test_retrieval.py — Testes de qualidade do retrieval e embeddings

Verifica:
    - Qualidade semântica (perguntas relevantes → chunks corretos)
    - Threshold de similaridade (fora do escopo → lista vazia)
    - Reranking (cross-encoder melhora ordenação)
    - Integridade do ChromaDB
    - Precision@k para perguntas conhecidas

NOTA: Carrega os modelos de embedding — mais lento que os outros testes.
      Os modelos de embedding e o Ollama cabem juntos na VRAM (RTX 4060 Ti 8 GB),
      portanto não é necessário forçar CPU via CUDA_VISIBLE_DEVICES.
"""

import pytest

from retrieval.search   import BuscadorRAG, SIMILARITY_THRESHOLD, TOP_K
from retrieval.reranker import RerankadorRAG, TOP_K_RERANKED


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def buscador():
    return BuscadorRAG()


@pytest.fixture(scope="module")
def rerankador():
    return RerankadorRAG()


# ── Testes de integridade do ChromaDB ─────────────────────────────────────────

def test_chromadb_conecta(buscador):
    """O ChromaDB deve conectar sem erros."""
    assert buscador.collection is not None


def test_chromadb_tem_chunks(buscador):
    """O ChromaDB deve ter chunks indexados."""
    assert buscador.collection.count() > 0, "ChromaDB vazio"


def test_chromadb_tem_quantidade_esperada(buscador):
    """Esperamos pelo menos 300 chunks dos 3 PDFs."""
    assert buscador.collection.count() >= 300


def test_chromadb_nome_colecao(buscador):
    """A coleção deve ter o nome correto."""
    assert buscador.collection.name == "ipardes_rag"


# ── Testes de qualidade semântica ─────────────────────────────────────────────

def test_busca_pib_retorna_desenvolvimento(buscador):
    """PIB → deve retornar chunks do documento de desenvolvimento paranaense."""
    resultados = buscador.buscar("crescimento do PIB do Paraná")
    assert len(resultados) > 0
    docs = {r.doc for r in resultados}
    assert "desenvolvimento_paranaense" in docs


def test_busca_emprego_2025_retorna_conjuntural(buscador):
    """Emprego 2025 → deve retornar chunks da análise conjuntural."""
    resultados = buscador.buscar("mercado de trabalho Paraná 2025")
    assert len(resultados) > 0
    docs = {r.doc for r in resultados}
    assert "analise_conjuntural" in docs


def test_busca_politicas_publicas_retorna_doc_correto(buscador):
    """Políticas públicas → deve retornar chunks do documento de avaliações."""
    resultados = buscador.buscar("avaliação de políticas públicas Brasil")
    assert len(resultados) > 0
    docs = {r.doc for r in resultados}
    assert "avaliacoes_politicas_publicas" in docs


def test_busca_expectativa_vida(buscador):
    """Expectativa de vida → deve retornar chunks do desenvolvimento paranaense."""
    resultados = buscador.buscar("expectativa de vida no Paraná")
    assert len(resultados) > 0
    docs = {r.doc for r in resultados}
    assert "desenvolvimento_paranaense" in docs


def test_busca_idh_parana(buscador):
    """IDH Paraná → deve retornar resultado relevante."""
    resultados = buscador.buscar("Índice de Desenvolvimento Humano Paraná")
    assert len(resultados) > 0


def test_busca_educacao_parana(buscador):
    """Educação → deve retornar resultado relevante."""
    resultados = buscador.buscar("educação e escolaridade no Paraná")
    assert len(resultados) > 0


# ── Testes de fora do escopo (críticos) ───────────────────────────────────────
#
# Testam o pipeline completo (buscador + rerankador), não só o buscador.
# Motivo: SIMILARITY_THRESHOLD é um pré-filtro permissivo — sua função é
# reduzir candidatos para o cross-encoder, não garantir relevância sozinho.
# A porta de qualidade final é o reranker (score > 0 do cross-encoder).
# Queries fora do escopo podem passar pelo threshold (especialmente com tabelas
# de baixa densidade), mas o cross-encoder as descarta com score negativo.

def test_fora_escopo_capital_franca(buscador, rerankador):
    """Capital da França → lista vazia após reranking (fora do escopo)."""
    chunks = buscador.buscar("Qual é a capital da França?")
    assert rerankador.rerankear("Qual é a capital da França?", chunks) == []


def test_fora_escopo_futebol(buscador, rerankador):
    """Futebol → lista vazia após reranking."""
    chunks = buscador.buscar("Quem ganhou o campeonato brasileiro de futebol?")
    assert rerankador.rerankear("Quem ganhou o campeonato brasileiro de futebol?", chunks) == []


def test_fora_escopo_presidente_brasil(buscador, rerankador):
    """Quem é o presidente → lista vazia após reranking."""
    chunks = buscador.buscar("Quem é o presidente do Brasil atualmente?")
    assert rerankador.rerankear("Quem é o presidente do Brasil atualmente?", chunks) == []


def test_fora_escopo_receita_culinaria(buscador, rerankador):
    """Receita culinária → lista vazia após reranking."""
    chunks = buscador.buscar("Como fazer bolo de chocolate?")
    assert rerankador.rerankear("Como fazer bolo de chocolate?", chunks) == []


def test_fora_escopo_historia_mundial(buscador, rerankador):
    """Segunda Guerra Mundial → lista vazia após reranking."""
    chunks = buscador.buscar("Quando terminou a Segunda Guerra Mundial?")
    assert rerankador.rerankear("Quando terminou a Segunda Guerra Mundial?", chunks) == []


# ── Testes de scores e threshold ─────────────────────────────────────────────

def test_scores_acima_do_threshold(buscador):
    """
    Todos os chunks retornados devem ter score >= SIMILARITY_THRESHOLD.
    Garante que o filtro está funcionando corretamente.
    """
    resultados = buscador.buscar("economia do Paraná")
    for r in resultados:
        assert r.score >= SIMILARITY_THRESHOLD, (
            f"Score abaixo do threshold: {r.score} < {SIMILARITY_THRESHOLD}"
        )


def test_scores_entre_zero_e_um(buscador):
    """Scores de similaridade de cosseno devem estar entre 0 e 1."""
    resultados = buscador.buscar("desenvolvimento econômico Paraná")
    for r in resultados:
        assert 0 <= r.score <= 1, f"Score fora do intervalo [0,1]: {r.score}"


def test_top_k_respeitado(buscador):
    """Busca nunca deve retornar mais que TOP_K chunks."""
    resultados = buscador.buscar("Paraná desenvolvimento")
    assert len(resultados) <= TOP_K


def test_resultado_tem_campos_obrigatorios(buscador):
    """Cada resultado deve ter doc, pagina, texto e score."""
    resultados = buscador.buscar("PIB Paraná crescimento")
    for r in resultados:
        assert r.doc
        assert r.pagina > 0
        assert r.texto.strip()
        assert r.score >= 0


def test_threshold_config_razoavel():
    """
    SIMILARITY_THRESHOLD deve estar entre 0.4 e 0.75.
    É apenas um pré-filtro grosseiro — o cross-encoder faz a filtragem fina.
    Muito baixo (< 0.4) = passa ruído demais pro reranker.
    Muito alto (> 0.75) = descarta chunks relevantes de documentos secundários
    em perguntas que abrangem múltiplos PDFs.
    """
    assert 0.4 <= SIMILARITY_THRESHOLD <= 0.75, (
        f"SIMILARITY_THRESHOLD fora do intervalo razoável: {SIMILARITY_THRESHOLD}"
    )


# ── Testes de precision@k ─────────────────────────────────────────────────────

@pytest.mark.parametrize("pergunta,doc_esperado", [
    ("crescimento PIB Paraná décadas",          "desenvolvimento_paranaense"),
    ("taxa desemprego Paraná 2025",             "analise_conjuntural"),
    ("metodologia avaliação política pública",  "avaliacoes_politicas_publicas"),
    ("expectativa vida nascimento Paraná",      "desenvolvimento_paranaense"),
    ("mercado trabalho emprego 2025 Paraná",    "analise_conjuntural"),
])
def test_precision_at_1(buscador, pergunta, doc_esperado):
    """
    Precision@1: o chunk mais relevante deve vir do documento esperado.
    Testa se o sistema recupera o documento correto para perguntas conhecidas.
    """
    resultados = buscador.buscar(pergunta)
    assert len(resultados) > 0, f"Nenhum resultado para: {pergunta}"
    melhor = max(resultados, key=lambda r: r.score)
    assert melhor.doc == doc_esperado, (
        f"Pergunta: '{pergunta}'\n"
        f"Esperado: {doc_esperado}, Obtido: {melhor.doc} (score={melhor.score})"
    )


# ── Testes de reranking ───────────────────────────────────────────────────────

def test_reranking_nao_aumenta_chunks(buscador, rerankador):
    """O reranking nunca deve aumentar o número de chunks."""
    pergunta    = "crescimento econômico Paraná"
    chunks      = buscador.buscar(pergunta)
    rerankeados = rerankador.rerankear(pergunta, chunks)
    assert len(rerankeados) <= len(chunks)


def test_reranking_respeita_top_k_reranked(buscador, rerankador):
    """O reranking deve retornar no máximo TOP_K_RERANKED chunks."""
    pergunta    = "crescimento econômico Paraná"
    chunks      = buscador.buscar(pergunta)
    rerankeados = rerankador.rerankear(pergunta, chunks)
    assert len(rerankeados) <= TOP_K_RERANKED


def test_reranking_descarta_chunks_negativos(buscador, rerankador):
    """Chunks com score negativo no cross-encoder devem ser descartados."""
    pergunta    = "crescimento econômico Paraná"
    chunks      = buscador.buscar(pergunta)
    rerankeados = rerankador.rerankear(pergunta, chunks)
    for chunk in rerankeados:
        assert chunk.score >= 0, f"Chunk negativo não descartado: {chunk.score}"


def test_reranking_lista_vazia(rerankador):
    """Reranking com lista vazia não deve lançar erros."""
    assert rerankador.rerankear("qualquer pergunta", []) == []


def test_reranking_chunk_mais_relevante_tem_score_positivo(buscador, rerankador):
    """O chunk top-1 após reranking deve ter score positivo."""
    pergunta    = "Qual foi o crescimento do PIB do Paraná?"
    chunks      = buscador.buscar(pergunta)
    rerankeados = rerankador.rerankear(pergunta, chunks)
    if rerankeados:
        assert rerankeados[0].score > 0


def test_reranking_ordenado_por_score(buscador, rerankador):
    """Chunks após reranking devem estar ordenados do maior pro menor score."""
    pergunta    = "desenvolvimento econômico Paraná"
    chunks      = buscador.buscar(pergunta)
    rerankeados = rerankador.rerankear(pergunta, chunks)
    if len(rerankeados) >= 2:
        for i in range(len(rerankeados) - 1):
            assert rerankeados[i].score >= rerankeados[i + 1].score, (
                f"Chunks fora de ordem: {rerankeados[i].score} < {rerankeados[i+1].score}"
            )


def test_reranking_preserva_metadados(buscador, rerankador):
    """Após reranking, doc e pagina de cada chunk devem ser preservados."""
    pergunta    = "PIB Paraná crescimento"
    chunks      = buscador.buscar(pergunta)
    rerankeados = rerankador.rerankear(pergunta, chunks)
    for chunk in rerankeados:
        assert chunk.doc
        assert chunk.pagina > 0
        assert chunk.texto.strip()


def test_reranking_fora_escopo_retorna_vazio(buscador, rerankador):
    """
    Para pergunta fora do escopo (sem chunks do buscador),
    o reranking deve retornar lista vazia.
    """
    chunks_vazios = buscador.buscar("Qual é a capital da França?")
    rerankeados   = rerankador.rerankear("Qual é a capital da França?", chunks_vazios)
    assert rerankeados == []
