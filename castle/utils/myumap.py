"""GPU-accelerated UMAP wrapper using RAPIDS cuML (optional)."""

try:
    from umap.spectral import spectral_layout
    from cuml.manifold.umap import fuzzy_simplicial_set, simplicial_set_embedding
    from cuml.manifold.umap_utils import find_ab_params
    from cuml.decomposition import PCA
    import cupy as cp
    import numpy as np

    class UMAP:
        """GPU-accelerated UMAP using RAPIDS cuML with spectral initialization.

        Combines cuML's fuzzy simplicial set and embedding optimization with
        CPU-side spectral layout from ``umap.spectral``. Falls back to PCA
        initialization when spectral layout is not available.

        Args:
            n_neighbors: Number of nearest neighbors for graph construction.
            n_components: Dimensionality of the output embedding.
            min_dist: Minimum distance between embedded points.
            n_epochs: Number of optimization epochs (default 20000).
            init: Initialization method ('spectral' or 'pca').
        """

        def __init__(self,  n_neighbors, n_components, min_dist=0.1, n_epochs=20000, init='spectral', random_state=np.random.randint(1, 1000), verbose=False):
            self.n_epochs = n_epochs
            self.n_neighbors = n_neighbors
            self.n_components = n_components
            self.min_dist = min_dist
            self.random_state = random_state
            self.verbose = verbose
            self.init = init


            
        def fit_transform(self, X):
            graph = fuzzy_simplicial_set(X, 
                                         n_neighbors=self.n_neighbors, 
                                         random_state=self.random_state, 
                                         metric='euclidean', 
                                         verbose=self.verbose)
            if self.init == 'spectral':
                layout = spectral_layout(X, graph.tocsr().get(), 
                                         dim=self.n_components, 
                                         random_state=self.random_state)
            elif self.init == 'pca':
                pca = PCA(n_components=self.n_components)
                X = cp.array(X)
                n_samples = len(X)
                selected = X.std(axis=0).argsort()[-n_samples+1:]
                layout = pca.fit_transform(X[:,selected])
            else:
                raise ValueError(f'Unknown init method: {self.init}')
                
            spread = 1.0
            a, b = find_ab_params(spread, self.min_dist)
            embedding = simplicial_set_embedding(X, graph, 
                                                 init=layout,
                                                 a=a, b=b,
                                                 n_epochs=self.n_epochs,
                                                 n_components=self.n_components,
                                                 random_state=self.random_state, 
                                                 verbose=self.verbose)
            if hasattr(embedding, 'to_host_array'):
                return embedding.to_host_array()
            else:
                return cp.asnumpy(embedding)

except ImportError:
    import warnings
    warnings.warn(
        "cuML not available. GPU-accelerated UMAP disabled. "
        "Install RAPIDS cuML for GPU UMAP support.",
        ImportWarning
    )
    raise ImportError(
        "castle.utils.myumap requires RAPIDS cuML. "
        "Install it with: pip install cuml-cu12 (or similar for your CUDA version). "
        "Falling back to CPU UMAP (umap-learn) is handled automatically."
    )
