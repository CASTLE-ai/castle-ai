import numpy as np

from sklearn.cluster import DBSCAN, HDBSCAN


import matplotlib.pyplot as plt
from matplotlib.patches import Patch


_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
_palette += ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']


def palette(x):
    if x == -1:
        return '#DDDDDD'
    return _palette[x % len(_palette)]


class Latent:
    def __init__(self, raw, time_window=1, device='cpu'):
        n = len(raw) // time_window
        num_feature = raw.shape[-1]
        self.time_window = time_window
        self.data = raw.reshape((-1,  num_feature * time_window))
        self.cluster = np.zeros(len(self.data)).astype(int)
        self.cluster[np.isnan(self.data.sum(axis=1))] = -1
        self.num_cluster = 1
        self.need_maintain_key_frames = True
        self.device=device

    def select(self, cluster_id):
        return LocalLatent(self.data[self.cluster == cluster_id], self.cluster == cluster_id, device=self.device)
    
    def merge(self, cluster_ids):
        cluster_ids = np.array(cluster_ids)
        mi = cluster_ids.min()

        for it in cluster_ids:
            self.cluster[self.cluster == it] = mi

        self.need_maintain_key_frames = True

    def maintain_key_frames(self):
        if hasattr(self, 'key_frames'):
            delattr(self, 'key_frames')
        n = len(self.data)
        self.key_frames = [0] + [i + 1 for i in range(n - 1) if self.cluster[i] != self.cluster[i + 1]] + [n - 1]
        self.need_maintain_key_frames = False

    def plot_syllables(self):
        if self.need_maintain_key_frames:
            self.maintain_key_frames()
            

        widths = [self.key_frames[j+1] - self.key_frames[j] for j in range(len(self.key_frames)-1)]
        colors = [palette(self.cluster[self.key_frames[j]]) for j in range(len(self.key_frames)-1)]
        lefts = self.key_frames[:-1]


        
        plt.bar(lefts, height=[1]*len(widths), width=widths, color=colors, align='edge', edgecolor='none')
        plt.xlim(0, self.key_frames[-1])
        plt.ylim(0, 1)
        plt.yticks([])
        unique_categories = sorted(set(self.cluster[self.key_frames[j]] for j in range(len(self.key_frames)-1)))
        legend_handles = [Patch(color=palette(cat), label=str(cat)) for cat in unique_categories]
        plt.legend(handles=legend_handles, title="Categories")


    def split_local_latent(self, local_latent):
        assert hasattr(local_latent, 'cluster')
        cluster = local_latent.cluster
        index_mask = local_latent.index_mask
        num_cluster_add_cluster = cluster.max() + 1
        

        old_cluster = self.cluster[index_mask]
        old_cluster[cluster == -1] = -1
        old_cluster[~(cluster == -1)] = cluster + self.num_cluster
        self.num_cluster += num_cluster_add_cluster
        self.cluster[index_mask] = old_cluster

        self.need_maintain_key_frames = True





class LocalLatent:
    def __init__(self, data, index_mask, device):
        self.data = data
        self.index_mask = index_mask
        self.device = device


        

    def build_embedding(self, configs):
        if self.device == 'cpu':
            from umap import UMAP
        elif self.device == 'gpu':
            from cuml.manifold import UMAP
        Z = self.data
        if hasattr(self, 'embedding'):
            delattr(self, 'embedding')

        if not type(configs) == list:
            configs = [configs]

        for it in configs:
            Z = UMAP(**it).fit_transform(Z)

        self.embedding = Z



    def build_cluster(self, method, configs):
        if self.device == 'cpu':
            from sklearn.cluster import DBSCAN, HDBSCAN
        elif self.device == 'gpu':
            from cuml.cluster import DBSCAN, HDBSCAN

        assert hasattr(self, 'embedding')
        if hasattr(self, 'cluster'):
            delattr(self, 'cluster')


        if method == 'hdbscan':
            self.cluster = HDBSCAN(**configs).fit_predict(self.embedding)
        elif method == 'dbscan':
            self.cluster = DBSCAN(**configs).fit_predict(self.embedding)
        else:
            assert False, f"method name should be hdbscan or dbscan, but got {method}."

    
    def plot_embedding(self, dims=[0, 1]):
        assert hasattr(self, 'embedding')
        assert len(dims) == 2, 'dims should'
        if hasattr(self, 'cluster'):
            for it in range(0, self.cluster.max()+1):
                plt.scatter(x=self.embedding[self.cluster == it,dims[0]], 
                            y=self.embedding[self.cluster == it,dims[1]], 
                            c=palette(it), 
                            label=f'{it}')
            if -1 in self.cluster:
                plt.scatter(x=self.embedding[self.cluster == -1,dims[0]], 
                            y=self.embedding[self.cluster == -1,dims[1]], 
                            c='grey',
                            label=f'-1')
            plt.legend()
        else:
            plt.scatter(x=self.embedding[:,dims[0]], 
                        y=self.embedding[:,dims[1]], 
                        c='grey')
            
    def merge(self, cluster_ids):
        cluster_ids = np.array(cluster_ids)
        mi = cluster_ids.min()

        for it in cluster_ids:
            self.cluster[self.cluster == it] = mi


            
        


