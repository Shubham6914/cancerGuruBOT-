from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from core.config import settings
from services.retrieval import QdrantHandlerLangChain

# Shared embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Qdrant client
client = QdrantClient(url=settings.qdrant_host)

# LangChain vector store wrapper
vector_store = QdrantVectorStore(
    client=client,
    collection_name=settings.qdrant_collection_name,
    embedding=embeddings
)

# Shared Qdrant handler
qdrant_handler = QdrantHandlerLangChain(
    vector_store=vector_store,
    client=client,
    embeddings=embeddings
)
