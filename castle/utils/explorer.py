import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_palette = [
    '#7AE4F0', '#FFD0EC', '#FBC471', '#6EE368', '#C1B5EA', '#A7CCED',
    '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', 
    '#FF6692', '#B6E880', '#FF97FF', '#FECB52', '#FD3216', '#00FE35', 
    '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', 
    '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', 
    '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', 
    '#BC7196', '#7E7DCD', '#FC6955', '#E48F72', '#66c5cc', '#f6cf71', 
    '#f89c74', '#dcb0f2', '#87c55f', '#9eb9f3', '#fe88b1', '#c9db74', 
    '#8be0a4', '#b497e7', '#b3b3b3', '#e58606', '#5d69b1', '#52bca3', 
    '#99c945', '#cc61b0', '#24796c', '#daa51b', '#2f8ac4', '#764e9f', 
    '#ed645a', '#a5aa99'
]


    
class Latent:
    def __init__(self, data, window=1):
        assert data.ndim == 2, 'Only design for (T x num_feats)'
        
        # Temporal concatenation
        n = (len(data) // window) * window
        num_feats = data.shape[-1]
        self.window = window
        self.data = data[:n].reshape((-1,  num_feats * window))
        

        self.syllables = np.zeros(len(self.data)).astype(int) - 1
        self.meta = []
        self.lookformeta = dict()
        
        # Init first cluster
        self.meta.append({
            'name': 'C0',
            'color': 'grey'
        })
        self.lookformeta['C0'] = 0
        self.syllables[~np.isnan(self.data.sum(axis=1))] = 0
        self.need_maintain_key_frames = True
        self.used_palette = set()
        
    
    def select(self, cid):
        # Generate FocusLatent
        export = np.copy(self.data)
        export[self.syllables != cid] = np.nan
        return FocusLatent(export, self.used_palette)
    
    def split(self, foucs_latent):
        self.need_maintain_key_frames = True
        focus = foucs_latent.focus
        focus_cluster = foucs_latent.cluster
        focus_cluster_set = set(focus_cluster[focus])
        if -1 in focus_cluster_set:
            focus_cluster_set.remove(-1)
        
        for it in focus_cluster_set:
            cid = len(self.meta)
            self.meta.append({
                'name': f'C{cid}',
                'color': foucs_latent.palette(it)
            })
            self.lookformeta[f'C{cid}'] = cid
            self.used_palette.add(foucs_latent.palette(it))
            mask = focus * (focus_cluster == it)
            self.syllables[mask] = cid
        
        
    def merge(self, cids):
        self.syllables[self.syllables == cid2] = cid1
        self.need_maintain_key_frames = True
        assert hasattr(self, 'syllables'), 'Do split first'
        assert len(cids) >= 2
        
        for i in range(1, len(cids)):
            self.syllables[self.syllables == cids[i]] = cids[0]
            
            
            
        
    def change_name(self, cid, name):
        assert not name in self.lookformeta, 'This name already be used.'
        self.meta[cid]['name'] = name
        return True
        
    def maintain_key_frames(self):
        self.need_maintain_key_frames = False
        n = len(self.data)
        self.key_frames = [0] + [i + 1 for i in range(n - 1) if self.syllables[i] != self.syllables[i + 1]] + [n - 1]
        
    def palette(self, c):
        if c >= 0 and c < len(self.meta):
            return self.meta[c]['color']
        else:
            return 'grey'
        
    def plot(self, legend=True):
        if self.need_maintain_key_frames:
            self.maintain_key_frames()

        widths, colors, lefts = [], [], []
        for j in range(len(self.key_frames) - 1):
            widths.append(self.key_frames[j + 1] - self.key_frames[j])
            colors.append(self.palette(self.syllables[self.key_frames[j]]))
            lefts.append(self.key_frames[j])

        # Plot the bar chart
        plt.bar(lefts, height=[1] * len(widths), width=widths, color=colors, align='edge', edgecolor='none')
        plt.xlim(0, self.key_frames[-1])
        plt.ylim(0, 1)
        plt.yticks([])

        # Prepare legend if necessary
        if legend:
            unique_categories = sorted(set(self.syllables[self.key_frames[j]] for j in range(len(self.key_frames) - 1)))
            if -1 in unique_categories:
                unique_categories.remove(-1)

            legend_handles = [
                Patch(color=self.palette(cat), label=self.meta[cat]['name'])
                for cat in unique_categories
            ]
            plt.legend(handles=legend_handles, title="Categories")


def gen_palette(avoid):
    res = [it for it in _palette if not it in avoid]
    return res

        
class FocusLatent:
    def __init__(self, data, color_avoid):
        self.data = data
        self.focus = (~np.isnan(self.data.sum(axis=1)))
        self.color_avoid = color_avoid
        self._palette = gen_palette(color_avoid)
        
        
    def palette(self, x):
        if x == -1:
            return 'grey'
        return self._palette[x % len(self._palette)]

        
    def gen_embedding(self, configs, device='cpu'):
        assert device == 'cpu' or device == 'gpu'
        if device == 'cpu':
            from umap import UMAP
        else:
            # from cuml.manifold import UMAP
            from myumap import UMAP
        
        if not type(configs) == list:
            configs = [configs]

        self.models = []
        Z = self.data[self.focus]
        for it in configs:
            model = UMAP(**it)
            Z = model.fit_transform(Z)
            self.models.append(model)

        self.embedding = np.zeros((len(self.data), Z.shape[-1])) + np.nan
        self.embedding[self.focus] = Z
        del UMAP
        
    def inference_embedding(self, data, device='cpu'):
        assert device == 'cpu' or device == 'gpu'
        if device == 'cpu':
            from umap import UMAP
        else:
            # from cuml.manifold import UMAP
            from myumap import UMAP

        Z = data
        for model in self.models:
            Z = model.transform(Z)
            
        del UMAP
        return Z
        
        
    def gen_cluster(self, config, device='cpu'):
        assert hasattr(self, 'embedding'), 'Do gen_embedding first'
        if device == 'cpu':
            from sklearn.cluster import DBSCAN
        else:
            from cuml.cluster import DBSCAN
            
        self.cluster = np.zeros(len(self.data)).astype(int) - 1
        C = DBSCAN(**config).fit_predict(self.embedding[self.focus])
        self.cluster[self.focus] = C
            
        
    def merge(self, cids):
        assert hasattr(self, 'cluster'), 'Do gen_cluster first'
        assert len(cids) >= 2
        
        for i in range(1, len(cids)):
            self.cluster[self.cluster == cids[i]] = cids[0]
        
        
    def plot(self, dimensions=[0, 1], legend=True):
        assert hasattr(self, 'embedding')
        assert len(dimensions) == 2
        if hasattr(self, 'cluster'):
            for it in range(0, self.cluster.max()+1):
                plt.scatter(x=self.embedding[self.cluster == it, dimensions[0]], 
                            y=self.embedding[self.cluster == it, dimensions[1]], 
                            c=self.palette(it), 
                            label=f'{it}')
            if -1 in self.cluster:
                
                plt.scatter(x=self.embedding[self.cluster == -1, dimensions[0]], 
                            y=self.embedding[self.cluster == -1, dimensions[1]], 
                            c='grey',
                            label=f'-1')
            if legend:
                plt.legend()
        else:
            plt.scatter(x=self.embedding[:, dimensions[0]], 
                        y=self.embedding[:, dimensions[1]], 
                        c='grey')