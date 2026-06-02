"""
tests/test_extract.py — Testes unitários do pré-processamento e limpeza de texto

Verifica:
    - Filtros de qualidade (pagina_e_valida)
    - Limpeza de texto (limpar_texto)
    - Integridade das páginas extraídas
    - Ausência de ruídos comuns (sumários, cabeçalhos, referências)
"""

import json
import pytest
from pathlib import Path
from indexing.extract import pagina_e_valida, limpar_texto, TITULOS_CABECALHO

PAGINAS_FILE = Path(__file__).parent.parent / "indexing" / "paginas_extraidas.json"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def paginas_reais():
    assert PAGINAS_FILE.exists(), "paginas_extraidas.json não encontrado"
    with open(PAGINAS_FILE, encoding="utf-8") as f:
        return json.load(f)


# ── Testes de pagina_e_valida ─────────────────────────────────────────────────

def test_pagina_valida_texto_normal():
    """Texto normal com conteúdo suficiente deve ser válido."""
    texto = "Este é um texto sobre desenvolvimento econômico do Paraná. " * 5
    assert pagina_e_valida(texto) is True


def test_pagina_invalida_texto_curto():
    """Textos com menos de 150 chars devem ser descartados (capas, separadores)."""
    assert pagina_e_valida("Texto curto.") is False


def test_pagina_invalida_texto_vazio():
    """String vazia deve ser inválida."""
    assert pagina_e_valida("") is False


def test_pagina_invalida_apenas_espacos():
    """Apenas espaços deve ser inválido."""
    assert pagina_e_valida("   \n\n   ") is False


def test_pagina_invalida_sumario():
    """Páginas com muitos '....' são sumários — devem ser descartadas."""
    texto = "Capítulo 1 ............. 10\nCapítulo 2 ............. 20\n" * 10
    assert pagina_e_valida(texto) is False


def test_pagina_invalida_referencias():
    """Páginas que começam com REFERÊNCIAS devem ser descartadas."""
    texto = "REFERÊNCIAS\n" + "AUTOR, N. Título do livro. Editora, 2020.\n" * 10
    assert pagina_e_valida(texto) is False


def test_pagina_invalida_ficha_tecnica():
    """Fichas técnicas com muitos ' - ' (cargo - nome) devem ser descartadas."""
    texto = "\n".join([f"Cargo {i} - Nome Pessoa {i}" for i in range(10)])
    assert pagina_e_valida(texto) is False


def test_pagina_valida_com_numeros():
    """Texto com dados numéricos e conteúdo econômico deve ser válido."""
    texto = (
        "O PIB do Paraná cresceu 2,5% ao ano nas últimas três décadas. "
        "A taxa de desemprego ficou em 5,2% no segundo trimestre de 2025. "
        "O setor industrial representa 30% da economia estadual. " * 3
    )
    assert pagina_e_valida(texto) is True


def test_pagina_valida_texto_tecnico():
    """Texto técnico de política pública deve ser considerado válido."""
    texto = (
        "A avaliação de impacto é o tipo mais frequente nas políticas públicas "
        "brasileiras, segundo o projeto de sistematização do IPARDES. "
        "Foram analisados 209 documentos entre 2014 e 2024. " * 3
    )
    assert pagina_e_valida(texto) is True


def test_pagina_limite_exato_150_chars():
    """Texto com exatamente 150 chars deve estar no limite — testar comportamento."""
    texto = "a" * 150
    # Independente do resultado (True ou False), não deve lançar exceção
    resultado = pagina_e_valida(texto)
    assert isinstance(resultado, bool)


def test_pagina_valida_nao_descarta_texto_com_traco():
    """Texto normal com alguns ' - ' mas abaixo do limiar não deve ser descartado."""
    texto = (
        "O setor de serviços - principal empregador - cresceu 3% em 2025. "
        "A indústria - segundo maior setor - manteve estabilidade. " * 5
    )
    # Tem 2 traços mas está longe do limiar de 6
    assert pagina_e_valida(texto) is True


# ── Testes de limpar_texto ────────────────────────────────────────────────────

def test_limpar_texto_remove_numero_pagina():
    """Linhas que são só um número devem ser removidas."""
    texto = "Conteúdo importante sobre o Paraná.\n42\nMais conteúdo relevante aqui."
    resultado = limpar_texto(texto)
    assert "42" not in resultado.split()


def test_limpar_texto_remove_titulo_cabecalho():
    """Títulos de cabeçalho repetidos devem ser removidos."""
    for titulo in TITULOS_CABECALHO:
        texto = f"{titulo}\nConteúdo da página com informações relevantes sobre economia."
        resultado = limpar_texto(texto)
        assert titulo not in resultado


def test_limpar_texto_preserva_conteudo():
    """O conteúdo relevante não deve ser removido pela limpeza."""
    conteudo = "O crescimento do PIB per capita do Paraná foi de 2,5% ao ano."
    resultado = limpar_texto(conteudo)
    assert "crescimento" in resultado
    assert "PIB"         in resultado
    assert "Paraná"      in resultado


def test_limpar_texto_normaliza_espacos():
    """Múltiplos espaços devem ser normalizados para um único espaço."""
    texto = "Texto   com    espaços   extras   no   meio."
    resultado = limpar_texto(texto)
    assert "  " not in resultado


def test_limpar_texto_string_vazia():
    """limpar_texto com string vazia deve retornar string vazia sem erros."""
    assert limpar_texto("") == ""


def test_limpar_texto_retorna_string():
    """limpar_texto deve sempre retornar uma string."""
    assert isinstance(limpar_texto("qualquer texto"), str)


def test_limpar_texto_remove_apenas_numeros_sozinhos():
    """
    Números que fazem parte de frases devem ser preservados.
    Só linhas que são exclusivamente um número devem ser removidas.
    """
    texto = "Em 2019, o PIB per capita cresceu 1,4%.\n42\nA taxa foi de 3%."
    resultado = limpar_texto(texto)
    assert "2019"  in resultado
    assert "1,4%"  in resultado
    assert "3%"    in resultado


def test_limpar_texto_multiplos_cabecalhos():
    """Múltiplos cabeçalhos na mesma página devem ser todos removidos."""
    texto = "\n".join(TITULOS_CABECALHO) + "\nConteúdo real da página econômica."
    resultado = limpar_texto(texto)
    for titulo in TITULOS_CABECALHO:
        assert titulo not in resultado
    assert "Conteúdo real" in resultado


# ── Testes das páginas extraídas reais ───────────────────────────────────────

def test_paginas_nao_vazias(paginas_reais):
    """O pipeline deve ter extraído pelo menos 100 páginas válidas."""
    assert len(paginas_reais) >= 100, f"Poucas páginas: {len(paginas_reais)}"


def test_paginas_campos_obrigatorios(paginas_reais):
    """Cada página deve ter doc, pagina e texto."""
    for pag in paginas_reais:
        assert "doc"    in pag
        assert "pagina" in pag
        assert "texto"  in pag


def test_paginas_sem_texto_vazio(paginas_reais):
    """Nenhuma página extraída deve ter texto vazio."""
    vazias = [p for p in paginas_reais if not p["texto"].strip()]
    assert len(vazias) == 0, f"{len(vazias)} páginas com texto vazio"


def test_paginas_sem_sumarios(paginas_reais):
    """Sumários (com muitos '....') devem ter sido filtrados."""
    com_sumario = [p for p in paginas_reais if p["texto"].count("....") > 5]
    assert len(com_sumario) == 0, f"{len(com_sumario)} páginas com padrão de sumário"


def test_paginas_sem_titulos_cabecalho(paginas_reais):
    """Títulos de cabeçalho devem ter sido removidos pelo limpar_texto."""
    amostra = paginas_reais[:30]
    for pag in amostra:
        for titulo in TITULOS_CABECALHO:
            assert titulo not in pag["texto"].upper(), (
                f"Título não removido em {pag['doc']} p.{pag['pagina']}"
            )


def test_tres_documentos_extraidos(paginas_reais):
    """Os três PDFs devem ter sido processados."""
    docs = {p["doc"] for p in paginas_reais}
    assert "desenvolvimento_paranaense"    in docs
    assert "analise_conjuntural"           in docs
    assert "avaliacoes_politicas_publicas" in docs


def test_paginas_texto_minimo(paginas_reais):
    """
    A grande maioria das páginas válidas deve ter pelo menos 150 chars.
    Toleramos até 2% de páginas curtas — podem ser legendas de figuras
    que passam pelo filtro inicial mas ficam curtas após a limpeza
    (ex: páginas com só "FONTE: ..." após remover o cabeçalho).
    """
    curtas = [p for p in paginas_reais if len(p["texto"]) < 150]
    proporcao = len(curtas) / len(paginas_reais)
    assert proporcao <= 0.02, (
        f"{len(curtas)} páginas abaixo de 150 chars ({proporcao:.1%} do total)"
    )


def test_paginas_numeracao_positiva(paginas_reais):
    """Números de página devem ser positivos."""
    invalidas = [p for p in paginas_reais if p["pagina"] <= 0]
    assert len(invalidas) == 0


def test_paginas_sem_referencias_bibliograficas(paginas_reais):
    """Páginas de referências bibliográficas devem ter sido filtradas."""
    refs = [p for p in paginas_reais if p["texto"].strip().upper().startswith("REFERÊNCIAS")]
    assert len(refs) == 0, f"{len(refs)} páginas de referências não filtradas"


def test_densidade_textual_razoavel(paginas_reais):
    """
    Texto médio por página deve ser razoável (> 300 chars).
    Muito abaixo indica que o filtro pode estar muito agressivo.
    """
    media = sum(len(p["texto"]) for p in paginas_reais) / len(paginas_reais)
    assert media > 300, f"Texto médio por página muito baixo: {media:.0f} chars"


def test_paginas_doc_e_string(paginas_reais):
    """O campo doc deve ser uma string não vazia."""
    for pag in paginas_reais:
        assert isinstance(pag["doc"], str) and pag["doc"].strip()
