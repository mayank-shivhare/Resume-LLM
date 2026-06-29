import streamlit as st
from src.rag_pipeline import build_resume_index, answer_resume_question, load_llm_pipeline

# Load resume index
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
llm_model = "google/flan-t5-large"
vector_store = build_resume_index(
    pdf_path="/Users/mayank/Antigtarvity_Projects/ResumeLLM/resume.pdf",  # Update with actual path
    embedding_model_name=embedding_model
)

llm_pipeline = load_llm_pipeline(llm_model)

# Streamlit UI
st.title("Resume Q&A")
question = st.text_input("Ask about your resume:")
if st.button("Get Answer"):
    if question:
        answer, _, _ = answer_resume_question(
            query=question,
            vector_store=vector_store,
            llm_pipeline=llm_pipeline
        )
        st.write("Answer:", answer)

# Instructions
st.write("Upload your resume PDF to complete setup")