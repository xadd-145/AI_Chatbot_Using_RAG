# app.py
import streamlit as st
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# === Configuration ===
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# === Basic PHI / sensitive info redaction ===
def redact_phi(text: str) -> str:
    """Mask emails and phone numbers inline."""
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(
        r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
        "[PHONE]",
        text,
    )
    return text


# === Cached resource loaders ===
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR, embedding_function=embeddings
    )
    # Retrieve top 3 docs only for faster context
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})


@st.cache_resource
def load_llm():
    # Streaming enabled for faster perceived response
    return ChatOllama(model="phi3:mini", temperature=0.1, streaming=True)


# === Generate answer using RAG ===
def generate_answer(query, retriever, llm):
    results = retriever.invoke(query)
    docs = results if isinstance(results, list) else [results]
    context = "\n\n".join([d.page_content for d in docs])
    context = context[:3000]  # Truncate for speed

    system_prompt = (
    "You are a precise assistant that answers questions ONLY using the provided context. "
    "Do not add explanations, examples, or definitions beyond what is explicitly mentioned in the context. "
    "If the user's question asks about specific terms, explain ONLY those terms if they exist in the context. "
    "If information is missing or not clearly available, respond with 'Not mentioned in the provided document.' "
    "Never speculate, repeat, or infer information beyond the provided text. "
    "Keep the answer concise and avoid repeating concepts."
    )


    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "Question: {q}\n\nContext:\n{ctx}")]
    )

    chain = prompt | llm

    # Stream tokens incrementally
    full_response = ""
    for chunk in chain.stream({"q": query, "ctx": context}):
        if chunk.content:
            full_response += chunk.content
            yield full_response


# === Streamlit UI ===
def main():
    st.set_page_config(page_title="PDF Chatbot (Local RAG)", page_icon="📄")
    st.title("📄 PDF Chatbot - Local RAG MVP")

    st.caption("Stack: Streamlit + Ollama (Phi-3 Mini) + Chroma  + PyPDF2 + NLTK")

    # Cache retriever + model only once
    if "retriever" not in st.session_state:
        st.session_state.retriever = load_retriever()
    if "llm" not in st.session_state:
        st.session_state.llm = load_llm()

    retriever = st.session_state.retriever
    llm = st.session_state.llm

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Initial assistant greeting
        st.session_state.chat_history.append(
            ("assistant", "👋 Hello Ms. Aditi! How can I help you today?")
        )

    # Display past messages
    for role, message in st.session_state.chat_history:
        st.chat_message(role).markdown(message)

    # Input field
    query = st.chat_input("Ask something about your PDF...")
    if query:
        query = redact_phi(query)
        st.chat_message("user").markdown(query)
        st.session_state.chat_history.append(("user", query))

        # Assistant streaming response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_text = ""
            try:
                for partial in generate_answer(query, retriever, llm):
                    full_text = partial
                    message_placeholder.markdown(full_text + "▌")
            except Exception as e:
                full_text = f"⚠️ Error: {e}"

            # Clean up and finalize output
            full_text = redact_phi(full_text)
            message_placeholder.markdown(full_text)
            st.session_state.chat_history.append(("assistant", full_text))


if __name__ == "__main__":
    main()
