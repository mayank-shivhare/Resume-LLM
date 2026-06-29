from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Create a HuggingFace embedding model instance."""
    return HuggingFaceEmbeddings(model_name=model_name)
