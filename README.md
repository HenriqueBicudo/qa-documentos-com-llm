# RAG IPARDES — Aprendizado de Máquina — Trabalho 2

API de chat baseada em RAG sobre documentos do IPARDES, 100% local.

## Requisitos de sistema

- Python 3.13+
- [Ollama](https://ollama.com/download) instalado e rodando
- Ghostscript instalado (para extração de tabelas com camelot)

## Instalação

```powershell
# 1. Instala PyTorch com suporte CUDA (GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Instala as dependências do projeto
pip install -r requirements.txt

# 3. Baixa o modelo LLM local (feito uma única vez)
ollama pull qwen2.5:7b-instruct
```

## Estrutura do projeto

```
├── data/                   # PDFs fonte do IPARDES
├── ingestion/              # Carregamento e extração de PDFs
│   ├── ingestor.py         # PDFLoader — texto + tabelas por página
│   └── table_extractor.py  # Localização e extração de tabelas (camelot)
├── indexing/               # Pipeline de indexação
│   ├── extract.py          # Extração e limpeza do texto dos PDFs
│   ├── chunking.py         # Divisão do texto em chunks
│   └── embed.py            # Geração de embeddings e indexação no ChromaDB
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

## Arquivos intermediários

Os arquivos `indexing/chunks.json` e `chroma_db/` já estão incluídos no ZIP
e podem ser usados diretamente, sem necessidade de regerar o banco de dados.

## Executar o pipeline de indexação (opcional — já gerado)

Só necessário se quiser regenerar o banco de dados do zero:

```powershell
# Extrai e limpa o texto dos PDFs
python -m indexing.extract

# Divide em chunks
python -m indexing.chunking

# Gera embeddings e indexa no ChromaDB (usa GPU se disponível)
python -m indexing.embed
```

## Executar a API e a interface

Abre **dois terminais** no diretório do projeto:

**Terminal 1 — API:**
```powershell
# CUDA_VISIBLE_DEVICES vazio: deixa a GPU exclusivamente para o Ollama
$env:CUDA_VISIBLE_DEVICES=""; uvicorn api.chat:app
```

**Terminal 2 — Interface:**
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
| qwen2.5:7b-instruct | 7.6B | Geração de respostas |

Todos os modelos rodam **100% localmente**, sem acesso à internet durante a execução.

## Executar os testes

```powershell
pytest tests/ -v
```
