"""
Embedding visualization utilities using UMAP/t-SNE.
"""

import numpy as np
import torch

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False


def reduce_embeddings(embeddings: dict, method="umap", n_components=2) -> dict:
    """
    Reduce embedding dimensionality for visualization.
    
    Args:
        embeddings: Dict of {name: tensor/array}
        method: "umap" or "tsne"
        n_components: Output dimensions (2 or 3)
        
    Returns:
        Dict of {name: [x, y, ...]}
    """
    if not embeddings:
        return {}
        
    names = list(embeddings.keys())
    vectors = []
    
    for name in names:
        emb = embeddings[name]
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        vectors.append(emb.flatten())
        
    X = np.array(vectors)
    
    # Need at least n_samples > n_components
    if len(X) < n_components + 1:
        # Just return first 2/3 dimensions
        reduced = X[:, :n_components]
    else:
        if method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(X) - 1),
                min_dist=0.1,
                metric='cosine'
            )
            reduced = reducer.fit_transform(X)
        elif method == "tsne" and TSNE_AVAILABLE:
            perplexity = min(30, len(X) - 1)
            reducer = TSNE(
                n_components=n_components,
                perplexity=max(5, perplexity),
                random_state=42
            )
            reduced = reducer.fit_transform(X)
        else:
            # PCA fallback
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(X)
    
    result = {}
    for i, name in enumerate(names):
        result[name] = reduced[i].tolist()
        
    return result


def create_plotly_scatter(reduced_embeddings: dict, title="Konuşmacı Embedding Uzayı"):
    """
    Create a Plotly scatter plot of embeddings.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
        
    if not reduced_embeddings:
        return None
        
    names = list(reduced_embeddings.keys())
    coords = list(reduced_embeddings.values())
    
    x = [c[0] for c in coords]
    y = [c[1] for c in coords]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        text=names,
        textposition="top center",
        marker=dict(
            size=15,
            color=list(range(len(names))),
            colorscale='Viridis',
            showscale=False
        ),
        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Boyut 1",
        yaxis_title="Boyut 2",
        template="plotly_dark",
        showlegend=False,
        height=500
    )
    
    return fig
