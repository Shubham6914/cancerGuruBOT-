 # Query embedding generation logic
from langchain_huggingface import HuggingFaceEmbeddings


def generate_embeddings(query: str) -> list[float]:
    """
    Generate embeddings for the given text using HuggingFace model.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings.embed_query(query)