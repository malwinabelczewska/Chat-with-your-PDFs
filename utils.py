import os
import numpy as np
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
    """Generate embeddings for text chunks using OpenAI's new API"""
    embeddings = []
    for chunk in chunks:
        try:
            response = client.embeddings.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error generating embedding for chunk: {e}")
            raise e

    return np.array(embeddings).astype("float32")


def search_similar_chunks(query, chunks, embeddings_or_index, top_k=3):
    """Search for similar chunks using either embeddings array or FAISS index"""
    try:
        # Generate query embedding
        query_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = query_response.data[0].embedding

        # Check if we're using FAISS index or regular embeddings
        if hasattr(embeddings_or_index, 'search'):  # It's a FAISS index
            index = embeddings_or_index
            distances, indices = index.search(
                np.array([query_embedding]).astype("float32"),
                top_k
            )
            results = [chunks[i] for i in indices[0]]
        else:  # It's a regular embeddings array
            embeddings = embeddings_or_index
            # Calculate cosine similarity
            query_vec = np.array(query_embedding).reshape(1, -1)
            similarities = np.dot(embeddings, query_vec.T).flatten()

            # Get top_k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            results = [chunks[i] for i in top_indices]

        return results

    except Exception as e:
        st.error(f"Error searching similar chunks: {e}")
        raise e


def answer_question_with_context(question, context_chunks):
    """Generate answer using GPT with context chunks"""
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return f"Sorry, I encountered an error while generating the answer: {str(e)}"