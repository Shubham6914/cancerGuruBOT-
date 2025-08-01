# Pydantic models for requests/responses
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    response: str
    