from RAG import retriever
from LLM_MODEL import LLM
from prompts import prompt
from langchain_core.messages import HumanMessage
import streamlit as st

st.title("ML Projects RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask about my ML projects...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.format(
        Context=context,
        question=question
    )

    response = LLM.invoke([
        HumanMessage(content=final_prompt)
    ])

    st.session_state.messages.append({"role": "assistant", "content": response.content})
    with st.chat_message("assistant"):
        st.write(response.content)
