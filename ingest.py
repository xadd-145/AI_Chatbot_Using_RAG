# ingest.py
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import nltk

# Download tokenizer if not already
nltk.download("punkt", quiet=True)

PDF_DIR = "data"
PERSIST_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def extract_text_from_pdfs(pdf_dir):
    """Extracts text from all PDFs in the 'data' folder using PyPDF2."""
    all_texts = []
    for file in os.listdir(pdf_dir):
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, file)
            print(f"📄 Extracting text from: {file}")
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            all_texts.append(text)
    if not all_texts:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")
    return all_texts

def split_text_into_chunks(texts):
    """Splits extracted text into small overlapping chunks using NLTK."""
    splitter = NLTKTextSplitter(chunk_size=800, chunk_overlap=120)
    docs = []
    for text in texts:
        docs.extend(splitter.create_documents([text]))
    print(f"✅ Created {len(docs)} text chunks for embedding.")
    return docs

def build_vector_db(docs):
    """Embeds chunks and stores them locally in Chroma vector DB."""
    print("⚙️ Building embeddings and saving to Chroma vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = Chroma.from_documents(
        docs, embedding=embeddings, persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    print(f"🎉 Vector store saved successfully in '{PERSIST_DIR}/'.")

if __name__ == "__main__":
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    texts = extract_text_from_pdfs(PDF_DIR)
    docs = split_text_into_chunks(texts)
    build_vector_db(docs)
