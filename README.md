# Resume RAG Chatbot

An open-source Resume RAG chatbot built with Streamlit, Hugging Face embeddings, and a local vector store.

## What it does
- Loads your resume PDF
- Splits it into text chunks
- Creates embeddings using Sentence Transformers
- Stores chunks in a Chroma vector database
- Answers your questions using an open-source Hugging Face LLM

## Requirements
- Python 3.11+
- A resume PDF at `data/resume.pdf`
- Optional: `HUGGINGFACE_HUB_TOKEN` for gated models

## Setup
1. Create a Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and update values:
   ```bash
   cp .env.example .env
   ```
3. Place your resume PDF at `data/resume.pdf`.

## Run
```bash
# Option 1: Simple Streamlit interface (recommended)
streamlit run main.py

# Option 2: Manual vector database creation
python3 build_vector_db.py

# Then run Streamlit
streamlit run main.py
```

## Project Status

### ✅ COMPLETED
- [x] Core RAG pipeline implementation
- [x] Document processor with PDF loading and chunking
- [x] Embedding generation using HuggingFace
- [x] Vector store with Chroma
- [x] LLM integration with HuggingFace
- [x] RAG orchestration
- [x] Streamlit UI with sleek, modern interface
- [x] Error handling and user experience
- [x] Environment configuration (.env)
- [x] Project structure and documentation
- [x] Manual vector database builder script

### 🔄 IN PROGRESS
- [ ] Phase 4: Deployment
  - [ ] GitHub repository setup
  - [ ] LICENSE file creation
  - [ ] Streamlit Cloud deployment
  - [ ] LinkedIn portfolio post

### 📁 FILES CREATED
- `main.py` - Streamlit app with sleek interface
- `src/document_processor.py` - PDF processing
- `src/embeddings.py` - Embedding generation
- `src/vector_store.py` - Vector database
- `src/llm_handler.py` - LLM integration
- `src/rag_pipeline.py` - RAG orchestration
- `README.md` - Documentation
- `.env` - Environment variables
- `.streamlit/config.toml` - Streamlit config
- `data/` - Resume placeholder directory
- `IMPLEMENTATION_PLAN.md` - Complete project plan
- `build_vector_db.py` - Manual vector database builder

### 🚀 READY TO RUN
```bash
# 1. Add your resume
mv your_resume.pdf data/resume.pdf

# 2. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Create vector database (optional but recommended)
python3 build_vector_db.py

# 4. Run the app
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
