# Should be replace by explorer.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import platform
OS_SYS = platform.uname().system
import torch

if OS_SYS == 'Darwin':
    DEFAULT_DEVICE = 'mps'
elif torch.cuda.is_available():
    DEFAULT_DEVICE = 'cuda'
else:
    DEFAULT_DEVICE = 'cpu'

_palette = [
    '#1f77b4', '#ffff33', '#1f78b4', '#ff7f00', '#17becf', '#e41a1c', '#c6dbef', '#e31a1c', '#9edae5', '#d62728', 
    '#dadaeb', '#2ca02c', '#fccde5', '#33a02c', '#f4cae4', '#393b79', '#ffffb3', '#f0027f', '#1b9e77', '#ff7f0e', 
    '#3182bd', '#ffd92f', '#386cb0', '#e6ab02', '#377eb8', '#e6550d', '#cbd5e8', '#d95f02', '#aec7e8', '#bf5b17', 
    '#e6f5c9', '#31a354', '#fff2ae', '#6a3d9a', '#ffed6f', '#5254a3', '#fed9a6', '#66a61e', '#decbe4', '#4daf4a', 
    '#f7b6d2', '#7b4173', '#f1e2cc', '#e7298a', '#ccebc5', '#fd8d3c', '#6baed6', '#fc8d62', '#6b6ecf', '#fdb462', 
    '#756bb1', '#fdbf6f', '#7570b3', '#ffbb78', '#66c2a5', '#fb8072', '#80b1d3', '#fdae6b', '#9ecae1', '#a55194', 
    '#c7e9c0', '#984ea3', '#fdd0a2', '#9467bd', '#a6d854', '#f781bf', '#74c476', '#e377c2', '#7fc97f', '#e78ac3', 
    '#b3de69', '#ce6dbd', '#98df8a', '#ff9896', '#8dd3c7', '#fb9a99', '#a6cee3', '#fdc086', '#8da0cb', '#fdcdac', 
    '#9c9ede', '#b5cf6b', '#de9ed6', '#b2df8a', '#bc80bd', '#b3e2cd', '#e7969c', '#b3cde3', '#fbb4ae', '#9e9ac8', 
    '#a1d99b', '#cab2d6', '#bcbddc', '#c5b0d5', '#bebada', '#beaed4'
]





def generate_palette(avoid):
    res = [it for it in _palette if not it in avoid]
    return res
    



class Latent:
    def __init__(self, raw, time_window=1, device=''):
        if len(device) == 0:
            device = DEFAULT_DEVICE
        n = (len(raw) // time_window) * time_window
        num_feature = raw.shape[-1]
        self.time_window = time_window
        self.data = raw[:n].reshape((-1,  num_feature * time_window))
        print('self.data:', self.data.shape)
        self.cluster = np.zeros(len(self.data)).astype(int)
        self.cluster[np.isnan(self.data.sum(axis=1))] = -1
        self.cluster_meta = dict()
        self.behavior_name2cluster_id = dict()
        
        self.cluster_meta[0] = {
            'name': 'root',
            'color': 'grey'
        }
        self.behavior_name2cluster_id['root'] = 0
        self.num_cluster = 1
        self.need_maintain_key_frames = True
        self.device=device
        
        self.used_palette = set()
        
    def get_time_window(self):
        return self.time_window

    def select(self, selected_cluster):
        if type(selected_cluster) == str:
            selected_cluster = self.behavior_name2cluster_id[selected_cluster]
        return LocalLatent(self.data[self.cluster == selected_cluster], self.cluster == selected_cluster, color_avoid=self.used_palette, device=self.device)
    
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

    def palette(self, c):
        if c in self.cluster_meta:
            return self.cluster_meta[c]['color']
        else:
            return 'grey'

    def plot_syllables(self):
        if self.need_maintain_key_frames:
            self.maintain_key_frames()
            

        widths = [self.key_frames[j+1] - self.key_frames[j] for j in range(len(self.key_frames)-1)]
        colors = [self.palette(self.cluster[self.key_frames[j]]) for j in range(len(self.key_frames)-1)]
        lefts = self.key_frames[:-1]


        
        plt.bar(lefts, height=[1]*len(widths), width=widths, color=colors, align='edge', edgecolor='none')
        plt.xlim(0, self.key_frames[-1])
        plt.ylim(0, 1)
        plt.yticks([])
        unique_categories = sorted(set(self.cluster[self.key_frames[j]] for j in range(len(self.key_frames)-1)))
        if -1 in unique_categories:
            unique_categories.remove(-1)

        legend_handles = [Patch(color=self.palette(cat), label=self.cluster_meta[cat]['name']) for cat in unique_categories]

        plt.legend(handles=legend_handles, title="Categories")



    def import_local_latent(self, local_latent):
        assert hasattr(local_latent, 'cluster')
        cluster = local_latent.cluster
        index_mask = local_latent.index_mask
        old_cluster = self.cluster[index_mask]

        # Check Name used?
        # for _, it in local_latent.export.items():
            # assert not it['name'] in self.behavior_name2cluster_id, 'new name be used'

        for cluster_local_id, it in local_latent.export.items():
            if it['name'] in self.behavior_name2cluster_id:
                continue
            cluster_id = self.num_cluster
            self.num_cluster += 1

            old_cluster[cluster == cluster_local_id] = cluster_id
            self.cluster_meta[cluster_id] = {
                'name': it['name'],
                'color': it['color']
            }
            self.behavior_name2cluster_id[it['name']] = cluster_id
            self.used_palette.add(it['color'])

        self.cluster[index_mask] = old_cluster

        self.need_maintain_key_frames = True


class LocalLatent:
    def __init__(self, data, index_mask, color_avoid, device):
        self.data = data
        self.index_mask = index_mask
        self.device = device
        self.color_avoid = color_avoid
        self._palette = generate_palette(color_avoid)

        self.export = dict()
        

    def build_embedding(self, configs):
        if self.device == 'cpu' or self.device == 'mps':
            from umap import UMAP
        elif 'cuda' in self.device:
            # try:
            #     from cuml.manifold import UMAP
            #     print("Using cuml.manifold.UMAP")
            # except:
                try:
                    from castle.utils.myumap import UMAP
                    print("Using castle.utils.myumap.UMAP")
                except:
                    from umap import UMAP
                    print("Using umap.UMAP")
        else:
            assert False, f'device error, expect cpu, mps, or cuda, got {self.device}'
        Z = self.data
        if hasattr(self, 'embedding'):
            delattr(self, 'embedding')

        if not type(configs) == list:
            configs = [configs]

        for it in configs:
            Z = UMAP(**it).fit_transform(Z)

        self.embedding = np.array(Z)
        self.configs = configs



    def build_cluster(self, method, configs):
        if self.device == 'cpu':
            from sklearn.cluster import DBSCAN

        elif 'cuda' in self.device:
            try:
                from cuml.cluster import DBSCAN
            except:
                from sklearn.cluster import DBSCAN

            

        assert hasattr(self, 'embedding')
        if hasattr(self, 'cluster'):
            delattr(self, 'cluster')


        if method == 'dbscan':
            self.cluster = DBSCAN(**configs).fit_predict(self.embedding)
        else:
            assert False, f"method name should be dbscan, but got {method}."

    def palette(self, x):
        if x == -1:
            return '#DDDDDD'
        return self._palette[x % len(self._palette)]

    
    def plot_embedding(self, dims=[0, 1]):
        assert hasattr(self, 'embedding')
        assert len(dims) == 2, 'dims should'

        embedding_data = self.embedding
        cluster_data = self.cluster if hasattr(self, 'cluster') else None

        if len(embedding_data) > 50000:
            idx = np.random.choice(len(embedding_data), 20000, replace=False)
            embedding_data = embedding_data[idx]
            if cluster_data is not None:
                cluster_data = cluster_data[idx]
        
        if cluster_data is not None:
            for it in range(0, cluster_data.max()+1):
                plt.scatter(x=embedding_data[cluster_data == it, dims[0]], 
                            y=embedding_data[cluster_data == it, dims[1]], 
                            c=self.palette(it), 
                            label=f'{it}')
            if -1 in cluster_data:
                plt.scatter(x=embedding_data[cluster_data == -1, dims[0]], 
                            y=embedding_data[cluster_data == -1, dims[1]], 
                            c='grey',
                            label=f'-1')
            plt.legend()
        else:
            plt.scatter(x=embedding_data[:, dims[0]], 
                        y=embedding_data[:, dims[1]], 
                        c='grey')
    
    def plot_name_embedding(self, dims=[0, 1]):
        assert hasattr(self, 'embedding')
        assert len(dims) == 2, 'dims should'
        if hasattr(self, 'cluster'):
            for it in range(0, self.cluster.max()+1):
                if it in self.export:
                    c = self.export[it]['color']
                    label = self.export[it]['name']
                else:
                    c = self.palette(-1)
                    label = it
                plt.scatter(x=self.embedding[self.cluster == it, dims[0]], 
                            y=self.embedding[self.cluster == it, dims[1]], 
                            c=c,
                            label=label)
           
           
            # for key, it in self.export.items():
            #     plt.scatter(x=self.embedding[self.cluster == key, dims[0]], 
            #                 y=self.embedding[self.cluster == key, dims[1]], 
            #                 c=it['color'],
            #                 label=it['name'])
            # for it in range(-1, self.cluster.max()+1):
            #     if it in self.export:
            #         continue
            #     plt.scatter(x=self.embedding[self.cluster == -1, dims[0]], 
            #                 y=self.embedding[self.cluster == -1, dims[1]], 
            #                 c='grey',
            #                 label=it)
            plt.legend()
        else:
            plt.scatter(x=self.embedding[:, dims[0]], 
                        y=self.embedding[:, dims[1]], 
                        c='grey')



    def merge(self, cluster_ids):
        cluster_ids = np.array(cluster_ids)
        mi = cluster_ids.min()

        for it in cluster_ids:
            self.cluster[self.cluster == it] = mi


    def label_cluster(self, cluster_id, cluster_name, cluster_color=''):
        tmp = dict()
        tmp['name'] = cluster_name
        tmp['color'] = cluster_color if len(cluster_color) > 0 else self._palette[cluster_id]
        # tmp['data'] = self.cluster == cluster_id

        self.export[cluster_id] = tmp
    
    def clean_label(self):
        self.export = dict()


