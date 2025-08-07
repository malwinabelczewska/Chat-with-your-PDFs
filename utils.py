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


def search_similar_chunks(query, chunks, search_target, top_k=3):
    """Search for similar chunks using ChromaDB collection"""
    try:
        # search_target is now the ChromaDB collection
        if hasattr(search_target, 'query'):  # It's a ChromaDB collection
            results = search_target.query(
                query_texts=[query],
                n_results=top_k
            )
            # Return the documents from ChromaDB results
            return results['documents'][0] if results['documents'] else []
        else:
            # Fallback for other cases (shouldn't happen with ChromaDB)
            st.error("Invalid search target")
            return []

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