import os
from typing import Optional

from src.document_processor import load_and_chunk_pdf
from src.embeddings import get_embedding_model
from transformers import pipeline

# Hugging Face LLM setup
def load_llm(model_name: str):
    return pipeline("text-generation", model=model_name)

# Update generate_response to use Hugging Face pipeline
def generate_response(llm_pipeline, prompt, max_new_tokens: int = 256):
    return llm_pipeline(prompt, max_new_tokens=max_new_tokens)
from src.vector_store import create_vector_store, load_vector_store


def build_resume_index(
    pdf_path: str,
    embedding_model_name: str,
    persist_directory: str = "vectordb",
):
    """Build a persistent vector index for a resume PDF."""
    chunks = load_and_chunk_pdf(pdf_path)
    embeddings = get_embedding_model(embedding_model_name)
    vector_store = create_vector_store(chunks, embeddings, persist_directory=persist_directory)
    return vector_store


def load_resume_index(
    embedding_model_name: str,
    persist_directory: str = "vectordb",
):
    """Load an existing resume vector index from disk."""
    embeddings = get_embedding_model(embedding_model_name)
    return load_vector_store(embeddings, persist_directory=persist_directory)


def resume_index_exists(persist_directory: str = "vectordb") -> bool:
    """Check whether the local vector store exists."""
    return os.path.isdir(persist_directory)


def make_resume_prompt(contexts: list[str], question: str, max_chars_per_chunk: int = 300) -> str:
    """Create a prompt that includes retrieved context and the user's question.

    Chunks are truncated to max_chars_per_chunk to ensure the full prompt
    (including the question) fits within flan-t5-large's 512-token input limit.
    """
    context_text = "\n\n".join(
        f"Excerpt {idx + 1}: {chunk[:max_chars_per_chunk]}"
        for idx, chunk in enumerate(contexts)
    )
    return (
        "Answer the question below using only the provided resume excerpts.\n\n"
        f"Resume excerpts:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer_resume_question(
    query: str,
    vector_store,
    llm_pipeline,
    k: int = 4,
    max_new_tokens: int = 256,
) -> tuple[str, list[str], str]:
    """Retrieve relevant resume chunks and generate an answer.

    Returns:
        (answer_text, context_chunks, inference_source)
    """
    results = vector_store.similarity_search(query, k=k)
    context_chunks = [result.page_content for result in results]
    prompt = make_resume_prompt(context_chunks, query)
    answer, source = generate_response(llm_pipeline, prompt, max_new_tokens=max_new_tokens)
    return answer, context_chunks, source


def load_llm_pipeline(model_name: str):
    """Load the text generation pipeline for the chosen LLM."""
    return load_llm(model_name)
