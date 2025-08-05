from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
load_dotenv()  # Load from .env file
import os
llm = ChatOllama(
    model=os.getenv("LLM_MODEL_NAME", "llama3"),
    temperature=0.5,
    base_url=os.getenv("LLM_API_URL", "http://localhost:11434").replace("/api/generate", "")
)

def call_llm_streaming(prompt: str):
    """
    Streams LLM response in real-time using a generator.
    
    Args:
        prompt (str): The user question/prompt.

    Yields:
        str: Chunks of the LLM-generated response.
    """
    try:
        messages = [HumanMessage(content=prompt)]
        stream = llm.stream(messages)  # ✅ Start streaming
    
        response_content = ""
        for chunk in stream:
            content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            response_content += content
            # Yield chunk as a JSON-like string (adjust as needed)
            yield json.dumps({"response": content, "done": False}) + "\n"
            print(f"LLM chunk: {content}")
        
        # Finalize the response
        print("LLM streaming completed." , response_content)

        # After streaming ends, you could yield a final done signal:
        yield json.dumps({"done": True}) + "\n"

    except Exception as e:
        print("LLM streaming error:", e)
        yield json.dumps({"response": "Sorry, an error occurred while generating the response.", "done": True}) + "\n"