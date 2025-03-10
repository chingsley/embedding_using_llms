import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]


def get_openai_code_embedding(code_snippet):
    """
    Get embeddings for code snippets using OpenAI's API.
    """
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=code_snippet,
            encoding_format="float",
            dimensions=3072  # Optional: maximum dimensions for more detailed code representation
        )

        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

