# Mayank's Resume Assistant 🤖

An open-source **Resume RAG Chatbot** — ask questions about Mayank's resume and get instant AI-powered answers.

Built with **Streamlit**, **HuggingFace Inference API**, **ChromaDB**, and **LangChain**.

---

## 🚀 Live Demo

[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://resume-llm.streamlit.app)

**[Click here to try it live!](https://resume-llm.streamlit.app)** — no installation needed.

---

## What it does

- Loads Mayank's resume PDF and splits it into searchable chunks
- Creates vector embeddings using Sentence Transformers
- Stores chunks in a Chroma vector database
- Answers your questions using **Llama-3.1-8B-Instruct** via HuggingFace

---

## 🛠️ Run Locally

### Requirements
- Python 3.11+
- A Hugging Face token (free) → [Get one here](https://huggingface.co/settings/tokens)

### Setup
```bash
# 1. Clone the repo
git clone https://github.com/mayank-shivhare/Resume-LLM.git
cd Resume-LLM

# 2. Create environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Set your HF token
cp .env.example .env
# Edit .env and add your HUGGINGFACE_HUB_TOKEN

# 4. Run
streamlit run main.py
```

The app auto-builds the vector database on first run.

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub and click **"New app"**
4. Select this repo, branch `ResumeLLM`, file `main.py`
5. In **"Advanced settings" → "Secrets"**, add:
   ```toml
   HUGGINGFACE_HUB_TOKEN = "your_hf_token_here"
   ```
6. Click **Deploy** 🚀

The app will auto-build the vector database on first startup.

---

## 📁 Project Structure

```
├── main.py                     # Streamlit app
├── build_vector_db.py          # Manual DB builder (optional)
├── src/
│   ├── document_processor.py   # PDF loading & chunking
│   ├── embeddings.py           # Embedding generation
│   ├── vector_store.py         # Chroma DB operations
│   ├── llm_handler.py          # HuggingFace Inference API
│   └── rag_pipeline.py         # RAG orchestration
├── data/Resume.pdf             # Mayank's resume
├── .streamlit/
│   ├── config.toml             # Streamlit theme config
│   └── secrets.toml            # Local secrets template
├── requirements.txt
├── .env                        # Local environment variables
└── .gitignore
```

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| PDF Parsing | PyMuPDF |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector DB | Chroma |
| LLM | `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Inference API |
| UI | Streamlit |

---

## 📊 Next Steps

- [x] Core RAG pipeline
- [x] HF Inference API integration
- [x] Streamlit UI
- [x] GitHub repository
- [ ] Streamlit Cloud deployment ⬅️
- [ ] LinkedIn portfolio post

---

## Technical Specifications

### Dependencies
```
streamlit==1.45.1
chromadb==0.6.3
sentence-transformers==5.0.0
PyMuPDF==1.26.0
langchain==0.3.25
langchain-community==0.3.24
langchain-text-splitters==0.3.8
transformers==4.52.4
torch==2.7.0
accelerate==1.7.0
python-dotenv==1.1.0
huggingface-hub==0.32.2
```

### Environment Variables
- `HUGGINGFACE_HUB_TOKEN` - HuggingFace authentication token
- `LLM_MODEL_ID` - HuggingFace model ID for LLM
- `EMBEDDING_MODEL_ID` - HuggingFace model ID for embeddings
- `RESUME_PDF_PATH` - Path to resume PDF
- `VECTORSTORE_DIR` - Directory for vector database

### Model Configuration
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM Model**: `google/flan-t5-large` (CPU-friendly)
- **Chunk Size**: 500 tokens
- **Chunk Overlap**: 50 tokens
- **Retrieval k**: 4 relevant chunks

### Performance Notes
- CPU-friendly model selection for local development
- Chroma vector database for fast similarity search
- Streamlit Cloud for free deployment
- HuggingFace Inference API for LLM inference

## Interface Features

### Simple, Sleek Design
- **Clean, modern UI** with responsive design
- **Streamlined chat interface** for easy interaction
- **Progress indicators** for better user experience
- **Source citations** for transparency

### Key Features
- **Instant answers** to resume-related questions
- **Semantic search** over resume content
- **Source context display** for verification
- **Error handling** with user-friendly messages
- **Status indicators** for vector database readiness

## Interface Usage

### Main Interface
1. **Status Check**: Sidebar shows vector database status
2. **Question Input**: Simple text input for asking questions
3. **Answer Display**: Clean, styled answer box with formatting
4. **Source Context**: Side panel showing relevant excerpts

### Workflow
1. **Setup**: Place resume PDF in `data/` directory
2. **Build Vector DB**: Use `build_vector_db.py` script (optional)
3. **Run App**: Start Streamlit interface
4. **Ask Questions**: Type questions and get instant answers
5. **Review Sources**: Check source context for verification

## LinkedIn Value Propositions
1. **Technical Skills**: RAG pipeline, embeddings, vector DB, LLMs, Streamlit
2. **Full Stack**: End-to-end ML system design
3. **Open Source**: Clean, documented, reusable code
4. **Deployment**: Live demo on Streamlit Cloud
5. **Self-Promotion**: Interactive resume that showcases technical abilities

## License
This project is open source.
