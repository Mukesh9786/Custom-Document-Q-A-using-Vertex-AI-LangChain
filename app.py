# vertex-doc-qa/main.py

import os
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle

PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

EMBEDDING_MODEL = "textembedding-gecko"
LLM_MODEL = "text-bison"

# 1. Load and extract text from PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = "\n".join(page.extract_text() for page in reader.pages)
    return text

# 2. Split text into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# 3. Generate embeddings using Vertex AI
def get_embeddings(text_chunks):
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    embeddings = [model.get_embeddings([chunk])[0].values for chunk in text_chunks]
    return np.array(embeddings, dtype='float32')

# 4. Store vectors in FAISS
INDEX_PATH = "vector.index"
META_PATH = "meta.pkl"

def store_embeddings(embeddings, text_chunks):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, 'wb') as f:
        pickle.dump(text_chunks, f)

# 5. Query the vector DB and return answer
def load_index_and_meta():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, 'rb') as f:
        texts = pickle.load(f)
    return index, texts

def query(question, top_k=3):
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    llm = TextGenerationModel.from_pretrained(LLM_MODEL)

    q_embedding = embed_model.get_embeddings([question])[0].values
    q_embedding = np.array([q_embedding], dtype='float32')

    index, texts = load_index_and_meta()
    distances, indices = index.search(q_embedding, top_k)
    context = "\n".join([texts[i] for i in indices[0]])

    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = llm.predict(prompt, temperature=0.3, max_output_tokens=256)
    return response.text

if __name__ == "__main__":
    pdf_path = "sample.pdf"  # Replace with your file
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("Splitting text...")
    chunks = split_text(text)

    print("Generating embeddings...")
    embeddings = get_embeddings(chunks)

    print("Storing in FAISS index...")
    store_embeddings(embeddings, chunks)

    print("Ready to ask questions! Type 'exit' to quit.")
    while True:
        q = input("\nAsk a question: ")
        if q.lower() == 'exit':
            break
        print("Answer:", query(q))
