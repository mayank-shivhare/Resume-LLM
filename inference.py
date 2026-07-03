#!/usr/bin/env python3
"""
Inference script for Resume RAG.

Usage:  python inference.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import load_resume_index, answer_resume_question


def main():
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model = os.getenv("LLM_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

    print("📂 Loading resume index...")
    vector_store = load_resume_index(embedding_model)

    print("✅ Ready! Type your questions below (or 'exit' to quit).\n")

    while True:
        query = input("💬 Question: ")
        if query.lower() in ("exit", "quit"):
            break

        answer, contexts, source = answer_resume_question(
            query=query,
            vector_store=vector_store,
            llm_pipeline=llm_model,
        )

        print(f"\n🤖 Answer: {answer}")
        print(f"📎 Source: {source}")
        print(f"📄 Contexts: {len(contexts)} chunks retrieved\n")


if __name__ == "__main__":
    main()