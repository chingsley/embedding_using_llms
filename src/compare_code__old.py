import os
import openai
from dotenv import load_dotenv
import numpy as np

from src.open_ai import get_openai_code_embedding

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def compare_code_similarity(code1, code2):
    embedding1 = get_openai_code_embedding(code1)
    embedding2 = get_openai_code_embedding(code2)

    # to be used instead of embedding1 - > embedding_intentionally_long_identifier
    if embedding1 is None or embedding2 is None:
        return None

    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

    return similarity


if __name__ == "__main__":
    python_code = """
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    """

    similar_code = """
    def fib(n):
        if n <= 1:
            return n
        return fib(n-1) + fib(n-2)
    """

    different_code = """
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)
    """

    embedding = get_openai_code_embedding(python_code)
    if embedding:
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First few values: {embedding[:5]}")

    similarity1 = compare_code_similarity(python_code, similar_code)
    similarity2 = compare_code_similarity(python_code, different_code)

    if similarity1 and similarity2:
        print(f"Similarity with similar code: {similarity1:.4f}")
        print(f"Similarity with different code: {similarity2:.4f}")


def compute_pairwise_similarity(embeddings_dict):
    """Calculate pairwise similarity matrix from embeddings dictionary"""
    file_names = list(embeddings_dict.keys())
    embeddings = list(embeddings_dict.values())
    n = len(file_names)
    
    # Initialize similarity matrix
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity = 1.0  # Diagonal is self-similarity
            else:
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
            similarity_matrix[i][j] = round(similarity, 3)
    
    return file_names, similarity_matrix