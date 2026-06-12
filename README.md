# RAG IPARDES — Aprendizado de Máquina — Trabalho 2

API de chat baseada em RAG sobre documentos do IPARDES, 100% local.

## Requisitos de sistema

- Python 3.13+
- [Ollama](https://ollama.com/download) instalado e rodando

## Instalação

```powershell
# 1. Instala PyTorch com suporte CUDA (GPU NVIDIA)
#    Verifique sua versão do CUDA com: nvidia-smi
#    e use a URL correspondente:
#      CUDA 12.8 → --index-url https://download.pytorch.org/whl/cu128
#      CUDA 12.4 → --index-url https://download.pytorch.org/whl/cu124
#      CUDA 12.1 → --index-url https://download.pytorch.org/whl/cu121
#      CUDA 11.8 → --index-url https://download.pytorch.org/whl/cu118
#    Exemplo para CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Instala as dependências do projeto
pip install -r requirements.txt

# 3. Baixa o modelo LLM local (feito uma única vez)
ollama pull qwen3:8b
```

## Estrutura do projeto

```
├── data/                   # PDFs fonte do IPARDES
├── ingestion/              # Carregamento e extração de PDFs
│   ├── ingestor.py         # PDFLoader — texto + tabelas por página
│   └── table_extractor.py  # [DEPRECADO] Implementação anterior com pdfplumber+camelot (histórico)
├── indexing/               # Pipeline de indexação
│   ├── extract.py          # Extração e limpeza do texto dos PDFs
│   ├── chunking.py         # Divisão do texto em chunks
│   ├── embed.py            # Geração de embeddings e indexação no ChromaDB
│   └── embed_tabelas.py    # Indexação das tabelas extraídas (mesmo ChromaDB)
├── retrieval/              # Busca semântica
│   ├── search.py           # Busca por similaridade de cosseno no ChromaDB
│   └── reranker.py         # Reranking com cross-encoder multilíngue
├── api/                    # API REST
│   ├── generator.py        # Pipeline RAG completo (busca → prompt → LLM)
│   └── chat.py             # Endpoints FastAPI
├── ui/                     # Interface de usuário
│   └── app.py              # Chat Streamlit
├── tests/                  # Testes unitários
├── indexing/chunks.json    # Chunks gerados (arquivo intermediário)
└── chroma_db/              # Banco vetorial persistido (arquivo intermediário)
```

## Banco de dados vetorial

O ChromaDB é usado no modo **embutido** (`PersistentClient`) — funciona como uma
biblioteca Python que persiste os dados em disco na pasta `chroma_db/`.
**Não requer Docker, servidor separado ou nenhuma instalação adicional.**

## Arquivos intermediários

Os arquivos abaixo já estão incluídos no ZIP e podem ser usados diretamente,
sem necessidade de regerar o banco de dados:

- `indexing/paginas_extraidas.json` — texto corrido por página após limpeza
- `indexing/tabelas_extraidas.json` — tabelas em markdown por página
- `indexing/chunks.json` — chunks gerados pelo chunking
- `chroma_db/` — banco vetorial com embeddings de texto e tabelas

## Executar o pipeline de indexação (opcional — já gerado)

Só necessário se quiser regenerar o banco de dados do zero:

```powershell
# Extrai e limpa o texto dos PDFs
python -m indexing.extract

# Divide em chunks
python -m indexing.chunking

# Gera embeddings e indexa o texto no ChromaDB (usa GPU se disponível)
python -m indexing.embed

# Indexa as tabelas extraídas na mesma coleção (executa em segundos)
python -m indexing.embed_tabelas
```

## Executar a API e a interface

Abre **três terminais** no diretório do projeto:

**Terminal 1 — Ollama (LLM local):**
```powershell
ollama serve
```
> Deixe este terminal aberto. Se o Ollama já estiver rodando em background, pode pular.

**Terminal 2 — API:**
```powershell
uvicorn api.chat:app
```
> Com 8 GB de VRAM todos os modelos cabem na GPU simultaneamente.
> Se sua GPU tiver menos de 6 GB, force os embeddings na CPU para liberar VRAM para o Ollama:
> - **Windows (PowerShell):** `$env:CUDA_VISIBLE_DEVICES=""; uvicorn api.chat:app`
> - **Linux/Mac:** `CUDA_VISIBLE_DEVICES="" uvicorn api.chat:app`

**Terminal 3 — Interface:**
```powershell
python -m streamlit run ui/app.py
```

Acessa a interface em: http://localhost:8501

A API REST fica disponível em: http://localhost:8000

## Endpoints da API

### POST /chat

Recebe uma pergunta e retorna **duas saídas** (exigência do professor):

```json
{
  "pergunta": "Qual foi o crescimento do PIB do Paraná?"
}
```

Resposta:
```json
{
  "pergunta": "...",
  "chunks_usados": [
    {"doc": "desenvolvimento_paranaense", "pagina": 7, "score": 3.83, "trecho": "..."}
  ],
  "prompt_final": "... prompt completo enviado à LLM ...",
  "resposta": "... resposta gerada pela LLM ..."
}
```

## Modelos utilizados

| Modelo | Parâmetros | Uso |
|--------|-----------|-----|
| intfloat/multilingual-e5-large | 560M | Embeddings (indexação e query) |
| cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | 117M | Reranking |
| qwen3:8b | 8B | Geração de respostas |

Todos os modelos rodam **100% localmente**, sem acesso à internet durante a execução.

## Executar os testes

```powershell
pytest tests/ -v
```
