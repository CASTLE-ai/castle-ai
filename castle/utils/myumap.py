from umap.spectral import spectral_layout
from cuml.manifold.umap import fuzzy_simplicial_set, simplicial_set_embedding
from cuml.manifold.umap_utils import find_ab_params
from cuml.decomposition import PCA
import cupy as cp
import numpy as np

class UMAP:
    def __init__(self,  n_neighbors, n_components, min_dist=0.1, n_epochs=10000, init='spectral', random_state=np.random.randint(1, 1000), verbose=False):
        print("here is mix UMAP")
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
            assert 0, 'init method error'
            
        
        spread = 1.0
        a, b = find_ab_params(spread, self.min_dist)
        embedding = simplicial_set_embedding(X, graph, 
                                             init=layout,
                                             a=a, b=b,
                                             n_epochs=self.n_epochs,
                                             n_components=self.n_components,
                                             random_state=self.random_state, 
                                             verbose=self.verbose)
        
        return embedding.to_host_array()