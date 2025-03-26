import requests
from chromadb import Client
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = Client()
collection = client.get_or_create_collection(name="rag_docs", embedding_function=embedding_func)

def retrieve_relevant_docs(query, k=3):
    results = collection.query(query_texts=[query], n_results=k)
    return results["documents"][0]

def query_llama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    try:
        data = response.json()
        if "response" not in data:
            print("⚠️ Ollama error:", data)
            return "Error: LLaMA generation failed."
        return data["response"]
    except Exception as e:
        print("❌ JSON decode error:", e)
        print("🔍 Raw response text:", response.text)
        return "Error: Failed to decode LLaMA response."


def generate_answer(user_query):
    context = retrieve_relevant_docs(user_query)
    full_prompt = f"""Answer the question based on the following context:\n\n{chr(10).join(context)}\n\nQuestion: {user_query}"""
    return query_llama(full_prompt)
