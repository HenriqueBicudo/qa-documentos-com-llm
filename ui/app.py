"""
ui/app.py — Interface de chat via Streamlit

Interface simples que consome a API FastAPI e exibe as duas saídas
exigidas pelo professor:
    - Resposta: texto gerado pela LLM para o usuário
    - Debug:    prompt completo + chunks usados (para avaliação)

Execute (com a API já rodando em outro terminal):
    streamlit run ui/app.py
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(
    page_title="RAG IPARDES",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG IPARDES")
st.caption("Chat baseado nos documentos do Instituto Paranaense de Desenvolvimento Econômico e Social")

# ── Histórico de mensagens ────────────────────────────────────────────────────

if "historico" not in st.session_state:
    st.session_state.historico = []

# ── Exibe histórico ───────────────────────────────────────────────────────────

for msg in st.session_state.historico:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input do usuário ──────────────────────────────────────────────────────────

pergunta = st.chat_input("Faça uma pergunta sobre os documentos do IPARDES...")

if pergunta:
    # Mostra a mensagem do usuário
    st.session_state.historico.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Chama a API
    with st.chat_message("assistant"):
        with st.spinner("Buscando nos documentos..."):
            try:
                response = requests.post(API_URL, json={"pergunta": pergunta}, timeout=120)
                response.raise_for_status()
                data = response.json()

                resposta     = data["resposta"]
                chunks       = data["chunks_usados"]
                prompt_final = data["prompt_final"]

                # ── Resposta principal ─────────────────────────────────────
                st.markdown(resposta)
                st.session_state.historico.append({"role": "assistant", "content": resposta})

                # ── Debug: trechos e prompt (exigência do professor) ───────
                with st.expander("Debug — Trechos usados e prompt final"):

                    if chunks:
                        st.subheader("Trechos recuperados")
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(
                                f"**[{i}] {chunk['doc']} — página {chunk['pagina']} "
                                f"(score: {chunk['score']})**"
                            )
                            st.text(chunk["trecho"][:400] + "..." if len(chunk["trecho"]) > 400 else chunk["trecho"])
                            st.divider()
                    else:
                        st.info("Nenhum trecho relevante encontrado.")

                    st.subheader("Prompt final enviado à LLM")
                    st.code(prompt_final, language="text")

            except requests.exceptions.ConnectionError:
                st.error("Não foi possível conectar à API. Certifique-se de que o uvicorn está rodando.")
            except Exception as e:
                st.error(f"Erro: {e}")