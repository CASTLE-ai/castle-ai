"""
castle/ui/embedding_scatter.py
EmbeddingScatterPlot — pure data/plotting class for embedding visualization.
No Gradio dependency.
"""

import io

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import KDTree

from castle.core.cluster import find_nearest_embedding


def padding(mi, mx, scale=1.05):
    mid = (mi + mx) / 2
    d = mx - mi
    return (mid - (d / 2) * scale), (mid + (d / 2) * scale)


class EmbeddingScatterPlot:
    """
    Handles plotting of embedding data and interaction (click to find nearest point).
    """
    def __init__(self, local_latents):
        data = local_latents.embedding
        self.local_latents = local_latents
        self.data = data
        
        # Calculate bounds once
        self.xlim = padding(data[:,0].min(), data[:,0].max())
        self.ylim = padding(data[:,1].min(), data[:,1].max())
        
        self.selected_point = (np.nan, np.nan)
        self.selected_index = -1
        
        # F-01: Build KDTree once and cache for reuse
        self._kdtree = KDTree(data)
    
    def pixel_2_embedding(self, px, py):
        px, py = float(px), float(py)
        # width/height are set by plot() — must be called first
        if not hasattr(self, 'width') or not hasattr(self, 'height'):
             raise RuntimeError('Plot not yet rendered. Call plot() first.')
             
        ex = (px / self.width) * (self.xlim[1] - self.xlim[0]) + self.xlim[0]
        ey = (py / self.height) * (self.ylim[1] - self.ylim[0]) + self.ylim[0]
        return ex, ey

    def plot(self):
        fig = plt.figure()
        self.local_latents.plot_embedding()
        plt.scatter(self.selected_point[0], self.selected_point[1], color='red')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim[1], self.ylim[0])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        self.width, self.height = img.size
        return img
    
    def plot_named_embedding(self):
        fig = plt.figure()
        self.local_latents.plot_name_embedding()
        plt.scatter(self.selected_point[0], self.selected_point[1], color='red')
        plt.axis('off')
        plt.xlim(self.xlim)
        plt.ylim(self.ylim[1], self.ylim[0])

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        self.width, self.height = img.size
        return img

    def save_named_embedding(self, save_path):
        index_mask = self.local_latents.index_mask
        masked_emb = self.local_latents.embedding
        masked_cls = self.local_latents.cluster
        config = self.local_latents.configs
        n_samples = len(index_mask)
        n_features = masked_emb.shape[-1]

        emb = np.zeros((n_samples, n_features)) + np.nan
        emb[index_mask] = masked_emb

        cls = np.zeros(n_samples, dtype=np.int16) - 1
        cls[index_mask] = masked_cls

        np.savez_compressed(save_path, emb=emb, cls=cls, config=config)
    
    def click(self, x, y):
        x, y = self.pixel_2_embedding(x, y)
        index = self.near_point(x, y)
        
        self.selected_point = self.data[index]
        self.selected_index = index # Local index in 'data'
        
        # Map back to global index
        self.selected_index = np.arange(len(self.local_latents.index_mask))[self.local_latents.index_mask][index]
        return self.plot()
        
    def near_point(self, x, y):
        # F-01: Use cached KDTree for O(log n) lookup
        index, _ = find_nearest_embedding(self.data, x, y, tree=self._kdtree)
        return index
