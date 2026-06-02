"""
api/generator.py — Geração de respostas via RAG + LLM local (Ollama)

Para cada pergunta:
    1. Busca os chunks relevantes no ChromaDB (retrieval)
    2. Monta o prompt com os trechos como contexto
    3. Chama o Qwen2.5:7b-instruct via Ollama (100% local)
    4. Retorna DUAS saídas:
        - debug: prompt completo + chunks usados
        - resposta: texto final gerado pela LLM

Execute para testar:
    python -m api.generator
"""

import requests

from dataclasses     import dataclass
from retrieval.search import BuscadorRAG, ResultadoBusca
from config.logger   import get_logger
from retrieval.reranker  import RerankadorRAG

logger = get_logger(__name__)

# ── Configurações ─────────────────────────────────────────────────────────────

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODELO_LLM  = "qwen2.5:7b-instruct"

# Temperatura baixa = respostas mais factuais, menos criativas
# 0.1 é conservador — reduz chance de alucinação
TEMPERATURE = 0.1

# Máximo de tokens na resposta — evita respostas infinitas
MAX_TOKENS  = 1024


# ── Templates de prompt ───────────────────────────────────────────────────────

PROMPT_COM_CONTEXTO = """Você é um assistente que responde perguntas com base em trechos de documentos do IPARDES.

REGRAS:
1. Use APENAS as informações presentes nos trechos abaixo para responder.
2. Cite sempre: (Documento, página X) após cada informação usada.
3. Se os trechos contiverem a informação, RESPONDA com ela — não diga que não está coberta.
4. Só diga "Esta informação não está coberta pelos documentos disponíveis." se os trechos realmente não contiverem nada relacionado à pergunta.
5. NUNCA invente dados, números ou fatos que não estejam nos trechos.

TRECHOS DOS DOCUMENTOS:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA (baseada apenas nos trechos acima):"""

PROMPT_SEM_CONTEXTO = """Esta informação não está coberta pelos documentos disponíveis.

Os documentos indexados tratam de desenvolvimento econômico e social do Paraná (IPARDES). Sua pergunta não encontrou trechos relevantes nesses documentos."""


# ── Tipos de saída ────────────────────────────────────────────────────────────

@dataclass
class RespostaRAG:
    """
    Saída completa do pipeline RAG para uma pergunta.

    Existem duas saídas separadas:
        debug:    prompt montado + chunks usados (para avaliação)
        resposta: texto final gerado pela LLM (para o usuário)
    """
    pergunta      : str
    chunks_usados : list[ResultadoBusca]
    prompt_final  : str
    resposta      : str


# ── Funções auxiliares ────────────────────────────────────────────────────────

def _formatar_contexto(chunks: list[ResultadoBusca]) -> str:
    """
    Formata os chunks recuperados em um bloco de contexto legível para a LLM.

    Cada trecho é apresentado com sua fonte (documento + página) para que
    a LLM possa citá-los na resposta.
    """
    partes = []
    for i, chunk in enumerate(chunks, 1):
        # Traduz o nome interno do doc para algo mais legível
        nome_doc = {
            "desenvolvimento_paranaense"   : "Desenvolvimento Paranaense",
            "analise_conjuntural"          : "Análise Conjuntural Jul/Ago 2025",
            "avaliacoes_politicas_publicas": "Avaliações de Políticas Públicas",
        }.get(chunk.doc, chunk.doc)

        partes.append(
            f"[Trecho {i} — {nome_doc}, página {chunk.pagina}]\n{chunk.texto}"
        )

    return "\n\n".join(partes)


def _chamar_ollama(prompt: str) -> str:
    """
    Envia o prompt para o Ollama e retorna a resposta gerada.

    Usa a API REST local do Ollama (http://localhost:11434).
    stream=False — espera a resposta completa antes de retornar.
    """
    payload = {
        "model" : MODELO_LLM,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama erro {response.status_code}: {response.text}")
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama não está rodando. Execute 'ollama serve' no terminal."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Timeout: o Ollama demorou mais de 120s para responder.")


# ── Pipeline principal ────────────────────────────────────────────────────────

class GeradorRAG:
    """
    Orquestra o pipeline completo: busca → prompt → geração → saída.
    """

    def __init__(self):
        self.buscador = BuscadorRAG()
        self.rerankador = RerankadorRAG()
    def responder(self, pergunta: str) -> RespostaRAG:
        """
        Gera uma resposta para a pergunta usando RAG.

        Fluxo:
            1. Busca chunks relevantes no ChromaDB
            2. Se não houver chunks: retorna mensagem de fora do escopo
            3. Se houver: monta prompt com contexto e chama a LLM
        """
        logger.info(f"Pergunta recebida: {pergunta}")

        # 1. Recupera chunks relevantes
        chunks = self.buscador.buscar(pergunta)
        if chunks:
            chunks = self.rerankador.rerankear(pergunta, chunks)
        logger.info(f"Chunks recuperados: {len(chunks)}")

        # 2. Sem contexto relevante — responde fora do escopo
        if not chunks:
            logger.info("Nenhum chunk relevante — respondendo fora do escopo")
            return RespostaRAG(
                pergunta     = pergunta,
                chunks_usados= [],
                prompt_final = PROMPT_SEM_CONTEXTO,
                resposta     = PROMPT_SEM_CONTEXTO,
            )

        # 3. Monta o prompt com os trechos recuperados
        contexto     = _formatar_contexto(chunks)
        prompt_final = PROMPT_COM_CONTEXTO.format(
            contexto=contexto,
            pergunta=pergunta,
        )

        # 4. Chama a LLM local
        logger.info("Chamando Ollama...")
        resposta = _chamar_ollama(prompt_final)
        # Pós-processamento: remove o fallback "não coberta" se a LLM já respondeu
        # Só remove se houver conteúdo real além da frase de fallback
        FRASE_FALLBACK = "Esta informação não está coberta pelos documentos disponíveis."
        if FRASE_FALLBACK in resposta:
            linhas = [l for l in resposta.split("\n") if FRASE_FALLBACK not in l]
            conteudo_restante = "\n".join(linhas).strip()
            # Só substitui se sobrou conteúdo real
            if conteudo_restante:
                resposta = conteudo_restante

        return RespostaRAG(
            pergunta     = pergunta,
            chunks_usados= chunks,
            prompt_final = prompt_final,
            resposta     = resposta,
        )


if __name__ == "__main__":
    gerador = GeradorRAG()

    perguntas = [
        "Qual foi o crescimento do PIB do Paraná?",
        "Qual é a capital da França?",
    ]

    for pergunta in perguntas:
        print(f"\n{'='*70}")
        resultado = gerador.responder(pergunta)

        print(f"PERGUNTA: {resultado.pergunta}")
        print(f"\n--- CHUNKS USADOS ({len(resultado.chunks_usados)}) ---")
        for c in resultado.chunks_usados:
            print(f"  [{c.doc} | p.{c.pagina} | score {c.score}] {c.texto[:100]}...")

        print(f"\n--- PROMPT FINAL ---\n{resultado.prompt_final[:500]}...")
        print(f"\n--- RESPOSTA ---\n{resultado.resposta}")