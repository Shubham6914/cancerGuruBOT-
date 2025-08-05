from .prompts import chat_prompt
from core.llm import call_llm

def run_rag_pipeline(query: str, context_docs: list) -> str:
    """
    Combine query and retrieved documents, build a prompt using chat format, call LLM, and return response.
    """
    
    print("\n📄 Retrieved Documents from Qdrant:")
    for i, doc in enumerate(context_docs):
        print(f"\n--- Document {i+1} ---\n{doc.page_content[:500]}")

    if not context_docs:
        print("No documents retrieved. Returning default response.")

    # Combine context from docs
    combined_context = "\n\n".join([doc.page_content for doc in context_docs])

    # Build chat-style prompt
    formatted_prompt = chat_prompt.format_messages(context=combined_context, user_query=query)

    print("\n🔍 LLM Prompt:")
    for msg in formatted_prompt:
        print(f"{msg.type.upper()}:\n{msg.content}\n")

    # Call your LLM (Ollama) with formatted chat messages
    prompt_str = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in formatted_prompt])
    response = call_llm(prompt_str)

    return response
