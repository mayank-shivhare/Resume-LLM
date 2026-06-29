from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings


def create_vector_store(
    text_chunks: list[str],
    embeddings: Embeddings,
    persist_directory: str = "vectordb",
) -> Chroma:
    """Create or persist a Chroma vector store from text chunks."""
    documents = [Document(page_content=chunk, metadata={"chunk_id": idx}) for idx, chunk in enumerate(text_chunks)]
    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="resume",
    )
    return vectordb


def load_vector_store(
    embeddings: Embeddings,
    persist_directory: str = "vectordb",
) -> Chroma:
    """Load an existing Chroma vector store from disk."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="resume",
    )
