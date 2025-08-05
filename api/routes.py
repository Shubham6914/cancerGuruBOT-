 # API endpoints (query endpoint)

from fastapi import FastAPI,APIRouter
from models.schemas import QueryRequest, QueryResponse
from services.embeddings import generate_embeddings
from services.retrieval import QdrantHandlerLangChain
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from core.config import settings
from services.vector_store_initializer import qdrant_handler, embeddings
from services.rag import run_rag_pipeline
from fastapi.responses import StreamingResponse
router = APIRouter()
@router.post("/query")
def user_query(payload: QueryRequest):
    """
    Process user query: generate embedding, retrieve context, run RAG pipeline, and return LLM response.
    """
    query = payload.query
    print(f"Received query: {query}")

    # Step 1: Generate embeddings for the query
    query_embeddings = generate_embeddings(query)
    print(f"Generated embeddings for query: {query_embeddings}")

    # Step 2: Retrieve top-k similar documents from Qdrant
    similar_docs = qdrant_handler.search_similar_documents_by_vector(query_embeddings, k=3)
    print(f"Retrieved {len(similar_docs)} similar documents.")

    # Step 3: Pass query and docs to RAG pipeline
    final_response = run_rag_pipeline(query, similar_docs)
    print(f"Final response from RAG pipeline: {final_response}")

    # Step 4: Return the final LLM response
    return StreamingResponse(
        final_response,
        media_type="application/json",  # or "text/event-stream" if using SSE
    )
