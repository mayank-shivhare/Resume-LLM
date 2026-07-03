#!/usr/bin/env python3

# Inference script for Resume RAG using Hugging Face Vector Store

import os
from src.rag_pipeline import load_resume_index, get_embedding_model
from src.embeddings import HuggingFaceVectorStore
from dotenv import load_dotenv
load_dotenv()

# Get Hugging Face API key from .env (fallback to empty string if not set)
HF_API_KEY = os.getenv("HF_API_TOKEN", "")

# Load existing vector store using the API key from environment
vector_db = HuggingFaceVectorStore(api_key=HF_API_KEY)

# Load resume index from existing store
index = load_resume_index(vector_db)

# Initialize RAG pipeline (adjust based on your implementation)
rag_pipeline = get_embedding_model()  # Or use your specific pipeline setup

# Inference loop
while True:
    query = input("Enter your resume query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break

    # Get relevant resume sections
    results = index.similarity_search(query)

    # Generate response using LLM
    response = rag_pipeline.generate_response(results)
    print("\n\"\"\"", response, "\"\"\"")

# Note: Ensure 'huggingface_hub' is in requirements.txt