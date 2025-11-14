# app.py
import streamlit as st
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# === Configuration ===
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === Basic PHI / sensitive info redaction ===
def redact_phi(text):
    """Mask emails and phone numbers inline."""
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(
        r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        "[PHONE]",
        text,
    )
    return text

# === Load vector DB ===
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR, embedding_function=embeddings
    )
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# === Load local LLM via Ollama ===
@st.cache_resource
def load_llm():
    return ChatOllama(model="mistral:7b", temperature=0.1)

# === Generate answer using RAG ===
def generate_answer(query, retriever, llm):
    # New retriever API uses .invoke(query)
    results = retriever.invoke(query)
    docs = results if isinstance(results, list) else [results]
    context = "\n\n".join([d.page_content for d in docs])


    system_prompt = (
        "You are a helpful assistant that answers ONLY using the provided context.\n"
        "If the answer is not in the context, say you don't know."
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "Question: {q}\n\nContext:\n{ctx}")]
    )

    chain = prompt | llm
    response = chain.invoke({"q": query, "ctx": context})
    return response.content

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="PDF Chatbot (Local RAG)", page_icon="📄")
    st.title("📄 PDF Chatbot - Local RAG MVP")

    if "AUTH_PASSWORD" in st.secrets:
        password = st.text_input("Enter password:", type="password")
        if password != st.secrets["AUTH_PASSWORD"]:
            st.stop()

    st.caption("Stack: Streamlit + Ollama (Llama 3) + Chroma + MiniLM + PyPDF2 + NLTK")

    retriever = load_retriever()
    llm = load_llm()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past messages
    for role, message in st.session_state.chat_history:
        st.chat_message(role).markdown(message)

    # Chat input
    query = st.chat_input("Ask something about your PDF...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.chat_history.append(("user", query))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    query = redact_phi(query)
                    answer = generate_answer(query, retriever, llm)
                    answer = redact_phi(answer)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))

if __name__ == "__main__":
    main()
