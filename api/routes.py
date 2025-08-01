 # API endpoints (query endpoint)

from fastapi import FastAPI,APIRouter
from models.schemas import QueryRequest, QueryResponse
from services.embeddings import generate_embeddings
from services.retrieval import QdrantHandlerLangChain
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from core.config import settings
from services.vector_store_initializer import qdrant_handler, embeddings

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def user_query(payload: QueryRequest) -> QueryResponse:
    """
    Process user query and return response.
    """
    query = payload.query
    # Generate embeddings for the query
    query_embeddings = generate_embeddings(query)

    # Initialize Qdrant handler (assuming vector_store is already set up

    similar_docs = qdrant_handler.search_similar_documents_by_vector(query_embeddings, k=3)
    # Here you would typically call your LLM API and process the response
    # For now, we will just return a mock response
    
    combined_content = "\n".join([doc.page_content for doc in similar_docs])

    return QueryResponse(response=combined_content)