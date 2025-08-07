import os
import faiss
import numpy as np
import openai
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def split_text_into_chunks(text, max_tokens=500, overlap=50):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    words = text.split()
    chunks = []
    chunk = []

    for word in words:
        chunk.append(word)
        tokens = tokenizer.encode(" ".join(chunk))
        if len(tokens) >= max_tokens:
            chunks.append(" ".join(chunk))
            chunk = chunk[-overlap:]

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def get_embeddings(chunks):
    # call OpenAI or other embedding provider
    embeddings = []
    for chunk in chunks:
        # Replace with actual embedding logic
        embedding = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
        embeddings.append(embedding)
    return np.array(embeddings).astype("float32")



def search_similar_chunks(query, chunks, chunk_embeddings, top_k=3):
    dimension = len(chunk_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(chunk_embeddings).astype("float32"))

    query_embedding = client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding

    distances, indices = index.search(np.array([query_embedding]).astype("float32"), top_k)
    results = [chunks[i] for i in indices[0]]
    return results


def answer_question_with_context(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
