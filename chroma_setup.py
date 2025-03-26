import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Folder with your PDFs
PDF_FOLDER = "data-processed"

# Function to extract text from a PDF
def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Initialize ChromaDB
embedding_func = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(name="rag_docs", embedding_function=embedding_func)

# Iterate over PDFs
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, filename)
        print(f"Processing: {filename}")
        text = extract_text_from_pdf(path)

        # Optional: Chunk text if too long (Chroma can limit per doc)
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"{filename}_{i}"]
            )

print("All documents loaded into ChromaDB.")
