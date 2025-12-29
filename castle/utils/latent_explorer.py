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
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
    '#393b79', '#5254a3', '#6b6ecf', '#9c9ede', '#637939', '#8ca252', '#b5cf6b', '#cedb9c', '#8c6d31', '#bd9e39',
    '#e7ba52', '#e7cb94', '#843c39', '#ad494a', '#d6616b', '#e7969c', '#7b4173', '#a55194', '#ce6dbd', '#de9ed6',
    '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2', '#31a354', '#74c476',
    '#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc', '#dadaeb', '#636363', '#969696', '#bdbdbd', '#d9d9d9',
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#bc80bd', '#ccebc5',
    '#ffed6f', '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6',
    '#6a3d9a', '#ffff99', '#b15928', '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d',
    '#666666', '#7fc97f', '#beaed4', '#fdc086', '#386cb0', '#f0027f', '#bf5b17', '#fbb4ae', '#b3cde3', '#decbe4',
    '#fed9a6', '#ffffcc', '#e5d8bd', '#fddaec', '#f2f2f2', '#b3e2cd', '#fdcdac', '#cbd5e8', '#f4cae4', '#e6f5c9',
    '#fff2ae', '#f1e2cc', '#cccccc', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ffff33', '#a65628', '#f781bf',
    '#999999', '#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'
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


