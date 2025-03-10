from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(embeddings, file_names, method):
    """
    Reduces embeddings to 2D using t-SNE and plots them with file names as labels.
    """
    embedding_vectors = np.array(list(embeddings.values()))
    n_samples = embedding_vectors.shape[0]
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embedding_vectors)

    plt.figure(figsize=(8, 6))
    for i, file_name in enumerate(file_names):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], label=file_name)
        plt.text(
            embeddings_2d[i, 0], 
            embeddings_2d[i, 1], 
            file_name, 
            fontsize=12,  # Increased from 9
            ha='center', 
            va='bottom'
        )
    plt.title("t-SNE Visualization of File Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.savefig(f"tsne_{method}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

def plot_pca(embeddings, file_names, method):
    """
    Reduces embeddings to 2D using PCA and plots them with file names as labels.
    """
    embedding_vectors = np.array(list(embeddings.values()))
    
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embedding_vectors)

    plt.figure(figsize=(8, 6))
    for i, file_name in enumerate(file_names):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], label=file_name)
        plt.text(
            embeddings_2d[i, 0], 
            embeddings_2d[i, 1], 
            file_name, 
            fontsize=12,  # Increased from 9
            ha='center', 
            va='bottom'
        )
    plt.title("PCA Visualization of File Embeddings")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.savefig(f"pca_{method}.pdf", dpi=300, bbox_inches='tight')
    plt.show()