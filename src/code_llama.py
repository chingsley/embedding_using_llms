import torch
from transformers import AutoTokenizer, AutoModel


def get_code_llama_embeddings(text, model_name="codellama/CodeLlama-7b-hf"):
    """
    Get embeddings for a given code using CodeLlama Python model.
    """
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Set to evaluation mode
        model.eval()

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Generate embeddings (without gradient calculation)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the last hidden state as embeddings
        # Mean pooling - take average of all token embeddings
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']

        # Multiply by attention mask to zero out padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                        min=1e-9)
        return embeddings[0]  # Return the first embedding (if batch size is 1)

    except Exception as e:
        print(f"Error getting CodeLlama embedding: {e}")
        return None


# Example usage
if __name__ == "__main__":
    sample_code = "def hello_world():\n    print('Namaste!')"
    embedding = get_code_llama_embeddings(sample_code)

    if embedding is not None:
        print(f"Successfully generated embedding with {embedding.shape[0]} dimensions")
        print(f"First few values: {embedding[:5].tolist()}")