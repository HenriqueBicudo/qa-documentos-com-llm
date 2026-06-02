"""
tests/test_generator.py — Testes do pipeline de geração de respostas

Verifica:
    - Construção correta do prompt (contexto + pergunta)
    - Formatação dos trechos (documento, página, texto)
    - Estrutura da RespostaRAG (campos obrigatórios)
    - Pós-processamento (remoção de frase de fallback duplicada)
    - Comportamento sem chunks (fora do escopo)

NOTA: Estes testes NÃO chamam o Ollama — testam apenas a lógica
      de montagem do prompt e formatação, sem depender da LLM.
"""

import pytest
from unittest.mock import patch, MagicMock
from retrieval.search   import ResultadoBusca
from api.generator import (
    _formatar_contexto,
    PROMPT_COM_CONTEXTO,
    PROMPT_SEM_CONTEXTO,
    GeradorRAG,
    RespostaRAG,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def chunk_pib():
    return ResultadoBusca(
        doc    = "desenvolvimento_paranaense",
        pagina = 7,
        texto  = "O PIB do Paraná cresceu 2,5% ao ano nas últimas décadas.",
        score  = 0.93,
    )


@pytest.fixture
def chunk_emprego():
    return ResultadoBusca(
        doc    = "analise_conjuntural",
        pagina = 12,
        texto  = "A taxa de desemprego no Paraná ficou em 5,2% no segundo trimestre de 2025.",
        score  = 0.91,
    )


@pytest.fixture
def chunk_politicas():
    return ResultadoBusca(
        doc    = "avaliacoes_politicas_publicas",
        pagina = 35,
        texto  = "Foram analisados 81 documentos nas áreas de saúde, educação e segurança.",
        score  = 0.90,
    )


# ── Testes de _formatar_contexto ──────────────────────────────────────────────

def test_formatar_contexto_um_chunk(chunk_pib):
    """Com um chunk, o contexto deve conter o texto e a referência correta."""
    contexto = _formatar_contexto([chunk_pib])
    assert "Desenvolvimento Paranaense" in contexto
    assert "página 7"                   in contexto
    assert "PIB do Paraná"              in contexto


def test_formatar_contexto_multiplos_chunks(chunk_pib, chunk_emprego):
    """Com múltiplos chunks, todos devem aparecer numerados."""
    contexto = _formatar_contexto([chunk_pib, chunk_emprego])
    assert "Trecho 1" in contexto
    assert "Trecho 2" in contexto
    assert "Desenvolvimento Paranaense"  in contexto
    assert "Análise Conjuntural"         in contexto


def test_formatar_contexto_tres_documentos(chunk_pib, chunk_emprego, chunk_politicas):
    """Chunks dos três documentos devem aparecer com nomes legíveis."""
    contexto = _formatar_contexto([chunk_pib, chunk_emprego, chunk_politicas])
    assert "Desenvolvimento Paranaense"       in contexto
    assert "Análise Conjuntural"              in contexto
    assert "Avaliações de Políticas Públicas" in contexto


def test_formatar_contexto_lista_vazia():
    """Contexto de lista vazia deve ser string vazia."""
    assert _formatar_contexto([]) == ""


def test_formatar_contexto_separador_entre_chunks(chunk_pib, chunk_emprego):
    """Os chunks devem ser separados por linhas em branco."""
    contexto = _formatar_contexto([chunk_pib, chunk_emprego])
    assert "\n\n" in contexto


def test_formatar_contexto_doc_desconhecido():
    """Documento com nome desconhecido deve usar o próprio nome como fallback."""
    chunk = ResultadoBusca(doc="doc_desconhecido", pagina=1, texto="Texto.", score=0.9)
    contexto = _formatar_contexto([chunk])
    assert "doc_desconhecido" in contexto


def test_formatar_contexto_preserva_texto_completo(chunk_pib):
    """O texto completo do chunk deve aparecer no contexto sem truncamento."""
    contexto = _formatar_contexto([chunk_pib])
    assert chunk_pib.texto in contexto


# ── Testes do template de prompt ─────────────────────────────────────────────

def test_prompt_com_contexto_tem_placeholders():
    """O template deve ter os placeholders {contexto} e {pergunta}."""
    assert "{contexto}" in PROMPT_COM_CONTEXTO
    assert "{pergunta}" in PROMPT_COM_CONTEXTO


def test_prompt_com_contexto_formatado(chunk_pib):
    """O prompt formatado deve conter o contexto e a pergunta."""
    contexto = _formatar_contexto([chunk_pib])
    prompt   = PROMPT_COM_CONTEXTO.format(
        contexto=contexto,
        pergunta="Qual foi o crescimento do PIB?",
    )
    assert "PIB do Paraná"               in prompt
    assert "Qual foi o crescimento"      in prompt
    assert "Desenvolvimento Paranaense"  in prompt


def test_prompt_sem_contexto_tem_mensagem_padrao():
    """O prompt de fora do escopo deve informar claramente a limitação."""
    assert "não está coberta" in PROMPT_SEM_CONTEXTO
    assert len(PROMPT_SEM_CONTEXTO) > 50


def test_prompt_com_contexto_tem_instrucao_citacao():
    """O prompt deve instruir a LLM a citar o documento e a página."""
    assert "página" in PROMPT_COM_CONTEXTO.lower() or "cite" in PROMPT_COM_CONTEXTO.lower()


def test_prompt_com_contexto_tem_instrucao_nao_inventar():
    """O prompt deve proibir explicitamente a invenção de informações."""
    assert "NUNCA" in PROMPT_COM_CONTEXTO or "não invente" in PROMPT_COM_CONTEXTO.lower()


# ── Testes do GeradorRAG (com mock do Ollama e dos modelos) ──────────────────

@pytest.fixture
def gerador_mockado():
    """
    Cria um GeradorRAG com buscador e rerankador mockados.
    Não carrega modelos reais — apenas testa a lógica de orquestração.
    """
    with patch("api.generator.BuscadorRAG") as MockBuscador, \
         patch("api.generator.RerankadorRAG") as MockReranker:

        mock_buscador  = MagicMock()
        mock_reranker  = MagicMock()
        MockBuscador.return_value  = mock_buscador
        MockReranker.return_value  = mock_reranker

        gerador = GeradorRAG()
        gerador.buscador   = mock_buscador
        gerador.rerankador = mock_reranker
        yield gerador, mock_buscador, mock_reranker


def test_gerador_retorna_fora_escopo_sem_chunks(gerador_mockado):
    """Quando não há chunks, a resposta deve indicar fora do escopo."""
    gerador, buscador, rerankador = gerador_mockado
    buscador.buscar.return_value   = []

    resultado = gerador.responder("Qual é a capital da França?")

    assert isinstance(resultado, RespostaRAG)
    assert resultado.chunks_usados == []
    assert "não está coberta" in resultado.resposta.lower()


def test_gerador_nao_chama_ollama_sem_chunks(gerador_mockado):
    """Sem chunks, o Ollama não deve ser chamado."""
    gerador, buscador, _ = gerador_mockado
    buscador.buscar.return_value = []

    with patch("api.generator._chamar_ollama") as mock_ollama:
        gerador.responder("Pergunta fora do escopo")
        mock_ollama.assert_not_called()


def test_gerador_chama_ollama_com_chunks(gerador_mockado, chunk_pib):
    """Com chunks relevantes, o Ollama deve ser chamado."""
    gerador, buscador, rerankador = gerador_mockado
    buscador.buscar.return_value    = [chunk_pib]
    rerankador.rerankear.return_value = [chunk_pib]

    with patch("api.generator._chamar_ollama", return_value="Resposta gerada.") as mock_ollama:
        resultado = gerador.responder("Qual foi o crescimento do PIB?")
        mock_ollama.assert_called_once()
        assert resultado.resposta == "Resposta gerada."


def test_gerador_prompt_contem_pergunta(gerador_mockado, chunk_pib):
    """O prompt enviado ao Ollama deve conter a pergunta do usuário."""
    gerador, buscador, rerankador = gerador_mockado
    buscador.buscar.return_value     = [chunk_pib]
    rerankador.rerankear.return_value = [chunk_pib]

    pergunta = "Qual foi o crescimento do PIB do Paraná?"

    with patch("api.generator._chamar_ollama", return_value="Resposta.") as mock_ollama:
        gerador.responder(pergunta)
        prompt_enviado = mock_ollama.call_args[0][0]
        assert pergunta in prompt_enviado


def test_gerador_prompt_contem_chunk(gerador_mockado, chunk_pib):
    """O prompt deve conter o texto do chunk recuperado."""
    gerador, buscador, rerankador = gerador_mockado
    buscador.buscar.return_value     = [chunk_pib]
    rerankador.rerankear.return_value = [chunk_pib]

    with patch("api.generator._chamar_ollama", return_value="Resposta."):
        resultado = gerador.responder("PIB do Paraná")
        assert chunk_pib.texto in resultado.prompt_final


def test_gerador_resultado_tem_campos_obrigatorios(gerador_mockado, chunk_pib):
    """RespostaRAG deve ter pergunta, chunks_usados, prompt_final e resposta."""
    gerador, buscador, rerankador = gerador_mockado
    buscador.buscar.return_value     = [chunk_pib]
    rerankador.rerankear.return_value = [chunk_pib]

    with patch("api.generator._chamar_ollama", return_value="Resposta."):
        resultado = gerador.responder("PIB?")
        assert resultado.pergunta
        assert resultado.prompt_final
        assert resultado.resposta
        assert isinstance(resultado.chunks_usados, list)


# ── Testes do pós-processamento ───────────────────────────────────────────────

def test_pos_processamento_remove_fallback_duplicado(gerador_mockado, chunk_pib):
    """
    Se a LLM retornar a frase de fallback junto com conteúdo real,
    a frase de fallback deve ser removida.
    """
    gerador, buscador, rerankador = gerador_mockado
    buscador.buscar.return_value     = [chunk_pib]
    rerankador.rerankear.return_value = [chunk_pib]

    resposta_llm = (
        "O PIB cresceu 2,5% ao ano.\n"
        "Esta informação não está coberta pelos documentos disponíveis."
    )

    with patch("api.generator._chamar_ollama", return_value=resposta_llm):
        resultado = gerador.responder("PIB?")
        assert "não está coberta" not in resultado.resposta
        assert "2,5%" in resultado.resposta


def test_pos_processamento_preserva_fallback_sozinho(gerador_mockado, chunk_pib):
    """
    Se a LLM retornar APENAS a frase de fallback (sem conteúdo real),
    ela deve ser mantida — é a resposta correta nesse caso.
    """
    gerador, buscador, rerankador = gerador_mockado
    buscador.buscar.return_value     = [chunk_pib]
    rerankador.rerankade.return_value = [chunk_pib]

    resposta_llm = "Esta informação não está coberta pelos documentos disponíveis."

    with patch("api.generator._chamar_ollama", return_value=resposta_llm):
        resultado = gerador.responder("PIB?")
        assert "não está coberta" in resultado.resposta
