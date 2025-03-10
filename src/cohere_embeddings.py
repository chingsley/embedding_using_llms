import cohere
import os

api_key = os.environ["COHERE_API_KEY"]

def generate_cohere_code_embedding(code_snippet, api_key = api_key):
    """
    Generates an embedding for a code snippet using Cohere's API.

    Parameters:
    code_snippet (str): The code snippet to generate an embedding for.
    api_key (str): Your Cohere API key.

    Returns:
    list: The embedding vector for the code snippet.
    """
    # Initialize the Cohere client
    co = cohere.Client(api_key)

    # Generate the embedding
    response = co.embed(
        texts=[code_snippet],
        model="embed-english-v3.0",  # Use the appropriate model for code embeddings
        input_type="clustering"  # Specify the input type (e.g., "clustering", "search_document", etc.)
    )

    # Extract the embedding vector
    embedding = response.embeddings[0]
    return embedding