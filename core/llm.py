from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()  # Load from .env file
import os
llm = ChatOllama(
    model=os.getenv("LLM_MODEL_NAME", "llama3"),
    temperature=0.5,
    base_url=os.getenv("LLM_API_URL", "http://localhost:11434").replace("/api/generate", "")
)
def call_llm(prompt: str) -> str:
    """
    Calls the local LLM using LangChain's ChatOllama and returns the response.

    Args:
        prompt (str): The full prompt including context and question.

    Returns:
        str: LLM-generated full response.
    """
    try:
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)

        # If response.content is a list, join it into a string
        if isinstance(response.content, list):
            return " ".join(str(part) for part in response.content).strip()
        
        # If it's a string already
        return response.content.strip()

    except Exception as e:
        print("LLM error:", e)
        return "Sorry, the LLM returned an invalid response."
