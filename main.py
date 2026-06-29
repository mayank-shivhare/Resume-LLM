import os
from pathlib import Path

import streamlit as st
import huggingface_hub as hf
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.rag_pipeline import (
    answer_resume_question,
    load_resume_index,
    load_llm_pipeline,
    resume_index_exists,
)


def get_settings() -> dict[str, str]:
    load_dotenv()
    return {
        "llm_model_id": os.getenv("LLM_MODEL_ID", "google/flan-t5-base"),
        "embedding_model_id": os.getenv(
            "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "persist_directory": os.getenv("VECTORSTORE_DIR", "vectordb"),
    }


def main():
    st.set_page_config(page_title="Resume RAG Chatbot", layout="centered")
    
    # Custom CSS for sleek design
    st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        padding: 12px 20px;
        font-size: 16px;
    }
    .stButton > button {
        border-radius: 25px;
        background-color: #4CAF50;
        color: white;
        padding: 12px 30px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .context-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 4px solid #4CAF50;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready { background-color: #4CAF50; }
    .status-loading { background-color: #FFA500; }
    .status-error { background-color: #f44336; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #2c3e50; font-size: 2.5rem;">Resume RAG Chatbot</h1>
        <p style="color: #7f8c8d; font-size: 1.2rem;">Ask questions about your resume and get instant answers</p>
    </div>
    """, unsafe_allow_html=True)
    
    settings = get_settings()
    
    # Status indicator
    st.sidebar.header("📊 Status")
    if resume_index_exists(settings["persist_directory"]):
        st.sidebar.markdown('<span class="status-indicator status-ready"></span> Vector Database Ready', unsafe_allow_html=True)
        resume_index = load_resume_index(
            settings["embedding_model_id"], persist_directory=settings["persist_directory"]
        )
    else:
        st.sidebar.markdown('<span class="status-indicator status-error"></span> Vector Database Not Ready', unsafe_allow_html=True)
        st.sidebar.error("Please build the vector database first")
        resume_index = None
    
    # Main chat interface
    if resume_index is not None:
        # No need to load pipeline for cloud inference
        pass
        
        # Question input
        st.markdown("### 💬 Ask Your Question")
        question = st.text_input(
            "What would you like to know about the resume?",
            placeholder="e.g., What are the candidate's skills?",
            key="question_input"
        )
        
        # Ask button with spinner
        if st.button("🔍 Ask Question", use_container_width=True):
            if question.strip():
                # Create progress container
                progress_container = st.container()
                
                with progress_container:
                    st.markdown("### 🔄 Processing Steps")
                    
                    # Step 1: Vector search
                    step1 = st.status("🔍 Searching vector database...", state="running", expanded=True)
                    with step1:
                        st.write("Finding relevant resume chunks...")
                        results = resume_index.similarity_search(question, k=4)
                        context_chunks = [result.page_content for result in results]
                        step1.update(label="✅ Vector search complete", state="complete", expanded=False)
                    
                    # Step 2: Building prompt
                    step2 = st.status("📝 Building prompt with context...", state="running", expanded=True)
                    with step2:
                        st.write(f"Found {len(context_chunks)} relevant chunks")
                        from src.rag_pipeline import make_resume_prompt
                        prompt = make_resume_prompt(context_chunks, question)
                        step2.update(label="✅ Prompt built successfully", state="complete", expanded=False)
                    
                    # Step 3: Generating answer
                    step3 = st.status("🤖 Generating answer with LLM...", state="running", expanded=True)
                    with step3:
                        st.write("Querying HuggingFace API (falling back to local inference if offline)...")
                        from src.llm_handler import generate_response
                        answer, inference_source = generate_response(settings["llm_model_id"], prompt, max_new_tokens=128)
                        step3.update(label=f"✅ Answer generated — {inference_source}", state="complete", expanded=False)
                
                # Create two columns for answer and context
                answer_col, context_col = st.columns([2, 1])
                
                with answer_col:
                    st.markdown("### 🤖 Answer")
                    
                    # Display answer with custom styling
                    st.markdown(f"""
                    <div class="answer-box">
                        <p style="margin: 0; font-size: 16px; line-height: 1.6;">{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Inference source badge
                    badge_color = "#2ecc71" if "Cloud" in inference_source else "#e67e22"
                    st.markdown(f"""
                    <div style="margin-top: 10px; display: inline-block;
                                background-color: {badge_color}; color: white;
                                padding: 4px 12px; border-radius: 20px;
                                font-size: 12px; font-weight: bold;">
                        {inference_source}
                    </div>
                    """, unsafe_allow_html=True)
                
                with context_col:
                    st.markdown("### 📚 Source Context")
                    if context_chunks:
                        st.markdown(f"**Found {len(context_chunks)} relevant excerpts:**")
                        for idx, chunk in enumerate(context_chunks, 1):
                            st.markdown(f"""
                            <div class="context-box">
                                <small><strong>Excerpt {idx}:</strong></small><br>
                                <small>{chunk[:150]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("*<small>No context sources found</small>*")
            else:
                st.warning("Please enter a question first!")
    
    # Sidebar with info
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    This app uses:
    • **Hugging Face** for LLM inference
    • **Chroma DB** for vector storage  
    • **LangChain** for RAG pipeline
    
    **Features:**
    • Semantic search over resume content
    • Instant answers with source citations
    • Clean, modern interface
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("📁 Make sure `data/resume.pdf` exists")
    st.sidebar.markdown("⚙️ Configure `.env` if needed")


if __name__ == "__main__":
    main()
