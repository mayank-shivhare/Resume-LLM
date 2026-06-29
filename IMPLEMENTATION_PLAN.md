# Resume RAG Chatbot - Implementation Plan

## Project Overview
Build an open-source RAG pipeline that allows users to chat with a resume using:
- Resume PDF chunking and embedding
- Vector database for semantic search
- Hugging Face open-source LLM
- Streamlit web interface
- Fully deployed online

---

## Technology Stack

### Document Processing
- **PyMuPDF** - PDF extraction and text parsing
- **LangChain** - Document chunking, RAG orchestration
- **RecursiveCharacterTextSplitter** - Smart text chunking

### Embeddings
- **Sentence Transformers** (HuggingFace) - `all-MiniLM-L6-v2` (lightweight, open-source)

### Vector Database
- **Chroma** - Simpler API than FAISS, fully open-source

### LLM
- **Hugging Face Models**:
  - `google/flan-t5-large` - CPU-friendly, good quality
  - `mistralai/Mistral-7B-Instruct-v0.2` - Excellent quality, needs GPU
  - `HuggingFaceH4/zephyr-7b-beta` - Excellent quality, needs GPU
- Use **HuggingFace Inference API** (free tier)

### Frontend & Deployment
- **Streamlit** - Simple web UI (fully open-source)
- **Streamlit Cloud** - Free hosting tier

### Project Structure
```
resume-rag-chatbot/
├── main.py                 # Streamlit app entry point
├── src/
│   ├── document_processor.py   # PDF loading & chunking
│   ├── embeddings.py           # Embedding generation
│   ├── vector_store.py         # Vector DB creation/search
│   ├── llm_handler.py          # LLM inference
│   └── rag_pipeline.py         # Core RAG logic
├── data/
│   └── resume.pdf          # User resume
├── requirements.txt        # Dependencies
├── .streamlit/config.toml  # Streamlit config
├── README.md               # Documentation
├── LICENSE                 # Open source license (MIT/Apache 2.0)
├── .gitignore
└── .env                    # Environment variables
```

---

## Project Phases

### Phase 1: Core RAG Pipeline (Local Development)
**Deliverables:**
- Document processor module
- Embedding generation
- Vector database setup
- RAG query pipeline

**Steps:**
1. Install dependencies (LangChain, HuggingFace, Chroma, etc.)
2. Create document processor to extract text from resume PDF
3. Implement chunking strategy (chunk_size=500, overlap=50)
4. Set up embeddings using SentenceTransformers
5. Build Chroma vector store
6. Implement RAG retrieval logic
7. Test locally with sample queries

### Phase 2: LLM Integration
**Deliverables:**
- LLM handler with HuggingFace models
- Prompt engineering for resume context
- Response generation

**Steps:**
1. Choose LLM (recommend: Mistral-7B or Zephyr-7B)
2. Set up HuggingFace Inference API or Ollama
3. Create system prompt template for resume context
4. Build prompt engineering pipeline
5. Integrate retrieval results into LLM prompts
6. Test end-to-end RAG pipeline

### Phase 3: Streamlit UI
**Deliverables:**
- Interactive web interface
- Chat history display
- Real-time streaming responses

**Steps:**
1. Create Streamlit app skeleton
2. Build chat interface (messages, input box)
3. Implement session state for chat history
4. Add query processing and response display
5. Add loading indicators and error handling
6. Style UI (optional: custom CSS)

### Phase 4: Production & Deployment
**Deliverables:**
- GitHub repository (fully open-source)
- Deployment on Streamlit Cloud
- Documentation & setup guide

**Steps:**
1. Create GitHub repository
2. Add comprehensive README with setup instructions
3. Add LICENSE (MIT recommended)
4. Create requirements.txt with pinned versions
5. Set up Streamlit Cloud deployment
6. Configure environment variables
7. Test deployed app
8. Add LinkedIn post with demo/documentation

---

## Key Technical Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Embeddings | all-MiniLM-L6-v2 | Fast, lightweight, good accuracy |
| Vector DB | Chroma | Simpler API, fully open-source |
| LLM | google/flan-t5-large | CPU-friendly, good quality |
| LLM Runtime | HF Inference API | Free tier, no server management |
| Frontend | Streamlit | Simple, Python-native, easy deployment |
| Hosting | Streamlit Cloud | Free tier perfect for portfolio project |

---

## Implementation Checklist

- [x] Phase 1: Core RAG Pipeline
  - [x] Document processor
  - [x] Embeddings setup
  - [x] Vector store creation
  - [x] Local testing
  
- [x] Phase 2: LLM Integration
  - [x] LLM handler setup
  - [x] Prompt engineering
  - [x] Response generation
  - [x] End-to-end testing
  
- [x] Phase 3: Streamlit UI
  - [x] Chat interface
  - [x] Session management
  - [x] Error handling
  - [x] UI polish
  
- [ ] Phase 4: Deployment
  - [ ] GitHub setup
  - [ ] Documentation
  - [ ] Streamlit Cloud deployment
  - [ ] LinkedIn portfolio post

---

## Timeline Estimate
- Phase 1: 2-3 hours
- Phase 2: 2-3 hours
- Phase 3: 1-2 hours
- Phase 4: 1 hour
- **Total: ~6-9 hours** (can be done in 1-2 days)

---

## LinkedIn Value Propositions
1. **Technical Skills**: RAG, embeddings, vector DB, LLMs, Streamlit
2. **Full Stack**: End-to-end ML system design
3. **Open Source**: Clean, documented, reusable code
4. **Deployment**: Live demo on Streamlit Cloud
5. **Self-Promotion**: Interactive resume that showcases skills

---

## Current Status (2026-06-26)

### ✅ COMPLETED
- [x] Core RAG pipeline implementation
- [x] Document processor with PDF loading and chunking
- [x] Embedding generation using HuggingFace
- [x] Vector store with Chroma
- [x] LLM integration with HuggingFace
- [x] RAG orchestration
- [x] Streamlit UI with chat interface
- [x] Error handling and user experience
- [x] Environment configuration (.env)
- [x] Project structure and documentation

### 🔄 IN PROGRESS
- [ ] Phase 4: Deployment
  - [ ] GitHub repository setup
  - [ ] LICENSE file creation
  - [ ] Streamlit Cloud deployment
  - [ ] LinkedIn portfolio post

### 📁 FILES CREATED
- `main.py` - Streamlit app
- `src/document_processor.py` - PDF processing
- `src/embeddings.py` - Embedding generation
- `src/vector_store.py` - Vector database
- `src/llm_handler.py` - LLM integration
- `src/rag_pipeline.py` - RAG orchestration
- `README.md` - Documentation
- `.env` - Environment variables
- `.streamlit/config.toml` - Streamlit config
- `data/` - Resume placeholder directory

### 🚀 READY TO RUN
```bash
# 1. Add your resume
mv your_resume.pdf data/resume.pdf

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the app
streamlit run main.py
```

### 📊 NEXT STEPS
1. Add LICENSE file (MIT recommended)
2. Push to GitHub repository
3. Deploy to Streamlit Cloud
4. Create LinkedIn portfolio post
5. Test deployed application

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