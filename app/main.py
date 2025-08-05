# FastAPI app entrypoint

from fastapi import FastAPI
from api.routes import router
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


app = FastAPI(
    title="My RAG API",
    description="A local RAG system powered by Llama and Qdrant",
    version="1.0.0"
)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include the API router
app.include_router(router)

# LLM initialization & prompt logic
app.add_event_handler("startup", lambda: print("Starting up the FastAPI app..."))
