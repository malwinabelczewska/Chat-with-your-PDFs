import os
import numpy as np
import tiktoken
import re
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from typing import List

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== ORIGINAL FUNCTIONS (Keep for backward compatibility) =====

def split_text_into_chunks(text, max_tokens=500, overlap=50):
    """Original chunking method - kept for compatibility"""
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

# ===== NEW SEMANTIC CHUNKING FUNCTIONS =====

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex patterns
    More sophisticated than simple period splitting
    """
    # Pattern to split on sentence endings while preserving abbreviations
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'

    sentences = re.split(sentence_pattern, text)

    # Clean up and filter out very short sentences
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    return sentences

def semantic_chunk_text(text: str, max_tokens: int = 500, min_tokens: int = 100) -> List[str]:
    """
    Split text at natural boundaries (paragraphs, sentences) while respecting token limits

    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk (prevents tiny fragments)

    Returns:
        List of semantically coherent chunks
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Step 1: Split into paragraphs (natural boundaries)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = len(tokenizer.encode(paragraph))

        # If single paragraph exceeds max_tokens, split it by sentences
        if para_tokens > max_tokens:
            # Split long paragraph into sentences
            sentences = split_into_sentences(paragraph)

            for sentence in sentences:
                sentence_tokens = len(tokenizer.encode(sentence))

                # If adding this sentence would exceed limit, finalize current chunk
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    if current_tokens >= min_tokens:  # Only add if chunk is substantial
                        chunks.append(' '.join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens

        # If adding this paragraph would exceed limit, finalize current chunk
        elif current_tokens + para_tokens > max_tokens and current_chunk:
            if current_tokens >= min_tokens:
                chunks.append(' '.join(current_chunk))
            current_chunk = [paragraph]
            current_tokens = para_tokens
        else:
            current_chunk.append(paragraph)
            current_tokens += para_tokens

    # Add final chunk if it exists and meets minimum size
    if current_chunk and current_tokens >= min_tokens:
        chunks.append(' '.join(current_chunk))

    return chunks

def enhanced_chunk_text(text: str, method: str = "semantic") -> List[str]:
    """
    Unified chunking function that lets you choose the method

    Args:
        text: Input text to chunk
        method: "original" or "semantic"

    Returns:
        List of text chunks
    """
    if method == "semantic":
        return semantic_chunk_text(text)
    else:
        return split_text_into_chunks(text)

# ===== EXISTING FUNCTIONS (unchanged) =====

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