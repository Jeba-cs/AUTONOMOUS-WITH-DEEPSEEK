# agent.py

from langchain.tools import Tool
from deepseek_wrapper import ask_deepseek
from vector_store import search_similar_chunks

def answer_question(query: str, vector_store) -> str:
    chunks = search_similar_chunks(vector_store, query, k=4)
    context = "\n\n".join([doc.page_content for doc in chunks])
    prompt = [
        {"role": "system", "content": "Answer the question using only the context. If no answer is found, say 'Not in PDF'."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    return ask_deepseek(prompt)

def summarize_pdf(vector_store) -> str:
    docs = vector_store.similarity_search("", k=20)
    full_text = "\n\n".join([doc.page_content for doc in docs])
    prompt = [
        {"role": "system", "content": "Summarize the following PDF content."},
        {"role": "user", "content": full_text[:8000]}
    ]
    return ask_deepseek(prompt)

def route_to_tool(user_query: str, vector_store) -> str:
    # Ask DeepSeek: Is this a question or summary request?
    tool_decision_prompt = [
        {"role": "system", "content": "Decide what the user wants. Respond only with 'question' or 'summary'."},
        {"role": "user", "content": user_query}
    ]
    tool_choice = ask_deepseek(tool_decision_prompt).strip().lower()

    if "summary" in tool_choice:
        return summarize_pdf(vector_store)
    
    # Otherwise assume it's a question
    answer = answer_question(user_query, vector_store)
    if "not in pdf" in answer.lower():
        return f"❌ No answer found in PDF. Would you like a summary instead?"
    return answer
