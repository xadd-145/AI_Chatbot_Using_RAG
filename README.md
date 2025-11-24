# PDF RAG Chatbot

A local RAG (Retrieval-Augmented Generation) chatbot that answers questions about your PDF documents using Streamlit, Ollama, and Chroma vector database.

## Stack
- **Streamlit** - Web interface
- **Ollama (Phi-3 Mini)** - Local LLM
- **Chroma** - Vector database
- **HuggingFace Embeddings** - Text embeddings
- **PyPDF2** - PDF text extraction
- **NLTK** - Text splitting

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running
   - Download from: https://ollama.ai
   - Make sure `phi3:mini` model is installed: `ollama pull phi3:mini`
3. **All Python packages** installed (see Installation below)

## Installation

1. Install Python dependencies:
   ```bash
   pip install --user -r requirements.txt
   ```

2. Make sure Ollama is running:
   ```bash
   ollama serve
   ```
   (Or ensure Ollama service is running in the background)

3. (Optional) If you need to ingest/update PDFs:
   - Place PDF files in the `data/` folder
   - Run: `python ingest.py`

## Running the Chatbot

**Simple command:**
```bash
python -m streamlit run app.py
```

**Or from the project directory:**
```bash
cd "c:\Coding Files\Aditi_Chatbots\V1\pdf-rag-mvp"
python -m streamlit run app.py
```

The app will start and automatically open in your browser at:
- **Local URL:** http://localhost:8501

## Quick Start Checklist

Before running, make sure:
- ✅ Ollama is installed and `phi3:mini` model is available (`ollama list`)
- ✅ Python packages are installed (`pip install --user -r requirements.txt`)
- ✅ Vector database exists in `chroma_db/` folder (run `python ingest.py` if needed)
- ✅ PDF files are in the `data/` folder (if you want to update the knowledge base)

## Troubleshooting

**"Unable to connect" error:**
- Make sure Ollama is running: `ollama serve`
- Check if port 8501 is available
- Verify all packages are installed: `pip install --user sentence-transformers langchain-chroma langchain-huggingface langchain-ollama`

**Import errors:**
- Install missing packages: `pip install --user <package-name>`
- Make sure you're in the project directory

**Vector database errors:**
- Run `python ingest.py` to create/update the vector database
- Make sure PDF files exist in the `data/` folder

## Project Structure

```
pdf-rag-mvp/
├── app.py              # Main Streamlit application
├── ingest.py           # PDF ingestion and vector DB creation
├── requirements.txt    # Python dependencies
├── data/              # PDF files directory
│   └── source.pdf
└── chroma_db/         # Vector database (auto-generated)
```
