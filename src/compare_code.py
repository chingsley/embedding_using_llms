import os
import openai
from dotenv import load_dotenv
import numpy as np

from src.open_ai import get_openai_code_embedding

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def calculate_similarity(embedding1, embedding2):
    """Core similarity calculation reused across functions"""
    if embedding1 is None or embedding2 is None:
        return None
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )

def compare_code_similarity(code1, code2):
    """Compare two code snippets via their embeddings"""
    embedding1 = get_openai_code_embedding(code1)
    embedding2 = get_openai_code_embedding(code2)
    return calculate_similarity(embedding1, embedding2)

def compute_pairwise_similarity(embeddings_dict):
    """Calculate pairwise similarity matrix from precomputed embeddings"""
    file_names = list(embeddings_dict.keys())
    embeddings = list(embeddings_dict.values())
    n = len(file_names)
    
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = round(
                calculate_similarity(embeddings[i], embeddings[j]) or 0, 
                3
            )
    # Ensure diagonal is exactly 1.0
    np.fill_diagonal(similarity_matrix, 1.0)
    return file_names, similarity_matrix