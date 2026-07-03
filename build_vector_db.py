#!/usr/bin/env python3
"""
Manual vector database builder for Resume RAG Chatbot.

This script allows you to manually create the vector database
without using the Streamlit interface.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on path so 'src.*' absolute imports work
_project_root = Path(__file__).parent.resolve()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.rag_pipeline import build_resume_index, load_resume_index, resume_index_exists
from src.embeddings import get_embedding_model
from src.document_processor import load_and_chunk_pdf


def main():
    print("🚀 Resume Vector Database Builder")
    print("=" * 50)
    
    # Check if resume exists
    pdf_path = "data/Resume.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ Error: Resume PDF not found at {pdf_path}")
        print("Please place your resume PDF in the data/ directory")
        return
    
    print(f"✅ Resume found: {pdf_path}")
    
    # Check if vector database already exists
    persist_directory = "vectordb"
    if resume_index_exists(persist_directory):
        print(f"⚠️  Vector database already exists at {persist_directory}")
        print("🗑️  Removing existing vector database...")
        import shutil
        shutil.rmtree(persist_directory)
    
    # Get embedding model
    print("\n📊 Setting up embedding model...")
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = get_embedding_model(embedding_model_name)
    print(f"✅ Using embedding model: {embedding_model_name}")
    
    # Load and chunk the PDF
    print("\n📄 Loading and chunking resume PDF...")
    chunks = load_and_chunk_pdf(pdf_path)
    print(f"✅ Created {len(chunks)} text chunks")
    print(f"   Chunk size: ~500 tokens, overlap: 50 tokens")
    
    # Build vector database
    print("\n🔍 Building vector database...")
    print("This may take a few minutes depending on PDF size...")
    
    try:
        vector_store = build_resume_index(
            pdf_path=pdf_path,
            embedding_model_name=embedding_model_name,
            persist_directory=persist_directory
        )
        
        print("✅ Vector database created successfully!")
        print(f"📁 Location: {os.path.abspath(persist_directory)}")
        print(f"📊 Number of chunks stored: {len(vector_store.get()['documents'])}")
        
        # Test the vector database
        print("\n🧪 Testing vector database...")
        test_results = vector_store.similarity_search("skills", k=2)
        print(f"✅ Similarity search works: found {len(test_results)} results")
        
        print("\n🎉 Vector database creation completed!")
        print("\nYou can now run the Streamlit app:")
        print("streamlit run main.py")
        
    except Exception as e:
        print(f"❌ Error creating vector database: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()