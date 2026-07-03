import os
from pathlib import Path

# Must be set before any tokenizer/model loads to avoid deadlock warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import (
    load_resume_index,
    resume_index_exists,
    build_resume_index,
)


def get_settings() -> dict[str, str]:
    load_dotenv()
    
    # Support both local .env and Streamlit Cloud secrets
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        try:
            hf_token = st.secrets.get("HUGGINGFACE_HUB_TOKEN")
            if hf_token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        except Exception:
            pass
    
    return {
        "llm_model_id": os.getenv("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"),
        "embedding_model_id": os.getenv(
            "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "persist_directory": os.getenv("VECTORSTORE_DIR", "vectordb"),
    }


def main():
    st.set_page_config(page_title="Mayank's Resume Assistant", layout="centered")
    
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
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 30px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 24px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .context-box {
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 4px solid #667eea;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready { background-color: #4CAF50; }
    .status-error { background-color: #f44336; }
    </style>
    """, unsafe_allow_html=True)
    
    settings = get_settings()
    
    # ===== SIDEBAR =====
    
    # 1. About (top of sidebar)
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <h3 style="color: #2c3e50;">👤 Mayank's Resume</h3>
        <p style="color: #7f8c8d; font-size: 0.9rem;">AI-powered resume Q&A</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # 2. Models in Use
    st.sidebar.markdown("### 🤖 Models in Use")
    st.sidebar.markdown(f"""
    **LLM:** `{settings['llm_model_id']}`
    
    **Embeddings:** `{settings['embedding_model_id']}`
    """)
    st.sidebar.markdown("---")
    
    # 3. Status
    st.sidebar.markdown("### 📊 Status")
    
    # Auto-build vector DB if it doesn't exist (needed for cloud deployment)
    if not resume_index_exists(settings["persist_directory"]):
        pdf_path = "data/Resume.pdf"
        if os.path.exists(pdf_path):
            with st.spinner("🔧 Building vector database from resume... This may take a minute."):
                try:
                    resume_index = build_resume_index(
                        pdf_path=pdf_path,
                        embedding_model_name=settings["embedding_model_id"],
                        persist_directory=settings["persist_directory"],
                    )
                    st.sidebar.markdown('<span class="status-indicator status-ready"></span> Vector Database Built ✅', unsafe_allow_html=True)
                except Exception as e:
                    st.sidebar.markdown('<span class="status-indicator status-error"></span> Build Failed ❌', unsafe_allow_html=True)
                    st.sidebar.error(f"Error building vector DB: {e}")
                    resume_index = None
        else:
            st.sidebar.markdown('<span class="status-indicator status-error"></span> Resume PDF Not Found ❌', unsafe_allow_html=True)
            st.sidebar.error(f"No resume found at `{pdf_path}`")
            resume_index = None
    else:
        st.sidebar.markdown('<span class="status-indicator status-ready"></span> Vector Database Ready ✅', unsafe_allow_html=True)
        resume_index = load_resume_index(
            settings["embedding_model_id"], persist_directory=settings["persist_directory"]
        )
    
    # 4. Source Context (collapsible, in sidebar) — will be populated after answering
    context_placeholder = st.sidebar.empty()
    
    # ===== MAIN AREA =====
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color: #2c3e50; font-size: 2.5rem;">Mayank's Resume Assistant</h1>
        <p style="color: #7f8c8d; font-size: 1.1rem;">Ask anything about Mayank's experience, skills, or background</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main chat interface
    if resume_index is not None:
        st.markdown("### 💬 Ask a Question")
        question = st.text_input(
            "What would you like to know about Mayank?",
            placeholder="e.g., What are Mayank's key skills?",
            key="question_input"
        )
        
        if st.button("🔍 Ask Question", use_container_width=True):
            if question.strip():
                progress_container = st.container()
                
                with progress_container:
                    st.markdown("### 🔄 Processing Steps")
                    
                    step1 = st.status("🔍 Searching vector database...", state="running", expanded=True)
                    with step1:
                        st.write("Finding relevant resume chunks...")
                        results = resume_index.similarity_search(question, k=4)
                        context_chunks = [result.page_content for result in results]
                        step1.update(label="✅ Vector search complete", state="complete", expanded=False)
                    
                    step2 = st.status("📝 Building prompt with context...", state="running", expanded=True)
                    with step2:
                        st.write(f"Found {len(context_chunks)} relevant chunks")
                        from src.rag_pipeline import make_resume_prompt
                        prompt = make_resume_prompt(context_chunks, question)
                        step2.update(label="✅ Prompt built successfully", state="complete", expanded=False)
                    
                    step3 = st.status("🤖 Generating answer with LLM...", state="running", expanded=True)
                    with step3:
                        st.write(f"Using model: `{settings['llm_model_id']}`")
                        from src.llm_handler import generate_response
                        answer, inference_source = generate_response(settings["llm_model_id"], prompt, max_new_tokens=128)
                        step3.update(label=f"✅ Answer generated — {inference_source}", state="complete", expanded=False)
                
                # Answer in main area (full width)
                st.markdown("### 🤖 Answer")
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
                            padding: 4px 14px; border-radius: 20px;
                            font-size: 12px; font-weight: bold;">
                    {inference_source}
                </div>
                """, unsafe_allow_html=True)
                
                # Source context in sidebar (collapsed by default)
                with context_placeholder.container():
                    with st.sidebar.expander("📚 Source Context", expanded=False):
                        if context_chunks:
                            st.markdown(f"Found {len(context_chunks)} relevant excerpts from Mayank's resume:")
                            for idx, chunk in enumerate(context_chunks, 1):
                                st.markdown(f"""
                                <div class="context-box">
                                    <small><strong>Excerpt {idx}:</strong></small><br>
                                    <small>{chunk[:200]}...</small>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("*No context sources found*")
            else:
                st.warning("Please enter a question first!")
    
    # Footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("📁 Resume: `data/resume.pdf`")


if __name__ == "__main__":
    main()
