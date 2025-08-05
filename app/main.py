# FastAPI app entrypoint

from fastapi import FastAPI
from api.routes import router
app = FastAPI()



app = FastAPI(
    title="My RAG API",
    description="A local RAG system powered by Llama and Qdrant",
    version="1.0.0"
)
# Include the API router
app.include_router(router)

# LLM initialization & prompt logic
app.add_event_handler("startup", lambda: print("Starting up the FastAPI app..."))
