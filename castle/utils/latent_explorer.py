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


_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
_palette += ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']
_palette += ['#66c5cc', '#f6cf71', '#f89c74', '#dcb0f2', '#87c55f', '#9eb9f3', '#fe88b1', '#c9db74', '#8be0a4', '#b497e7', '#b3b3b3']
_palette += ['#e58606', '#5d69b1', '#52bca3', '#99c945', '#cc61b0', '#24796c', '#daa51b', '#2f8ac4', '#764e9f', '#ed645a', '#a5aa99']





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
        self.cluster = np.zeros(len(self.data)).astype(int)
        self.cluster[np.isnan(self.data.sum(axis=1))] = -1
        self.cluster_meta = dict()
        self.behavior_name2cluster_id = dict()
        
        self.cluster_meta[0] = {
            'name': 'init',
            'color': 'grey'
        }
        self.behavior_name2cluster_id['init'] = 0
        self.num_cluster = 1
        self.need_maintain_key_frames = True
        self.device=device
        
        self.used_palette = set()
        
        

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
        for _, it in local_latent.export.items():
            assert not it['name'] in self.behavior_name2cluster_id, 'new name be used'

        for cluster_local_id, it in local_latent.export.items():
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
            from cuml.manifold import UMAP
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



    def build_cluster(self, method, configs):
        if self.device == 'cpu':
            from sklearn.cluster import DBSCAN
            from hdbscan import HDBSCAN
        elif 'cuda' in self.device:
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

    def palette(self, x):
        if x == -1:
            return '#DDDDDD'
        return self._palette[x % len(self._palette)]

    
    def plot_embedding(self, dims=[0, 1]):
        assert hasattr(self, 'embedding')
        assert len(dims) == 2, 'dims should'
        if hasattr(self, 'cluster'):
            for it in range(0, self.cluster.max()+1):
                plt.scatter(x=self.embedding[self.cluster == it, dims[0]], 
                            y=self.embedding[self.cluster == it, dims[1]], 
                            c=self.palette(it), 
                            label=f'{it}')
            if -1 in self.cluster:
                plt.scatter(x=self.embedding[self.cluster == -1, dims[0]], 
                            y=self.embedding[self.cluster == -1, dims[1]], 
                            c='grey',
                            label=f'-1')
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


