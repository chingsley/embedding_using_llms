from src.hf_embeddings import get_hf_embeddings
from src.open_ai import get_openai_code_embedding
from src.cohere_embeddings import generate_cohere_code_embedding
from src.utils import read_files_from_folder
from src.compare_code import compute_pairwise_similarity
from src.plot import plot_tsne, plot_pca
from typing import Callable
import argparse, os
import numpy as np

EMBEDDINGS_METHODS = {
    'hf': get_hf_embeddings,
    'openai': get_openai_code_embedding,
    'cohere': generate_cohere_code_embedding,
}

def generate_embeddings(file_contents: dict, method: Callable, hf_model: str = None):
    """
    Generates embeddings for the given file contents using the specified method.
    """
    embeddings = {}
    for file_name, content in file_contents.items():
        print(f"\tGenerating embeddings for {file_name}...")
        if hf_model:
            embeddings[file_name] = method(content, hf_model)
        else:
            embeddings[file_name] = method(content)

    return embeddings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate embeddings for code snippets.')
    parser.add_argument('--method', choices=EMBEDDINGS_METHODS.keys(), help='Embedding method to use.')
    parser.add_argument('--plot_type', choices=['tsne', 'pca'], help='Plot type to use.')
    parser.add_argument('--hf_model', required=False ,type=str, help='HuggingFace model tag.')
    args = parser.parse_args()
    
    folders = {
        "Logger": os.path.join("data", "Logger"),
        "PersonObject": os.path.join("data", "PersonObject")
    }

    # Perform checks
    if args.method == 'hf' and not args.hf_model:
        parser.error('HuggingFace model tag is required for hf method.')

    method = EMBEDDINGS_METHODS[args.method]

    all_embeddings = {}
    for folder_name, folder_path in folders.items():
        print(f"Generating embeddings for files in {folder_name}:")
        file_contents = read_files_from_folder(folder_path)
        if args.method == 'hf':
            embeddings = generate_embeddings(file_contents, method, args.hf_model)
        else:
            embeddings = generate_embeddings(file_contents, method)
        all_embeddings.update(embeddings)

    file_names, similarity_matrix = compute_pairwise_similarity(all_embeddings)
    print("Pairwise Similarity Matrix:")
    # print(similarity_matrix.tostring(float_format="%.3f"))
    print(np.array2string(similarity_matrix, 
                    formatter={'float_kind': lambda x: "%.3f" % x},
                    threshold=np.inf))  # Prevents array truncation
    
    # Plot visualization
    plot_fn = plot_tsne if args.plot_type == 'tsne' else plot_pca
    plot_fn(all_embeddings, file_names, args.method)