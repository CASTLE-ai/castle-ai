import numpy as np

from castle.core.config import PALETTE_HEX

_palette = PALETTE_HEX * 5




    
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
        assert hasattr(self, 'syllables'), 'Do split first'
        assert len(cids) >= 2
        
        for i in range(1, len(cids)):
            self.syllables[self.syllables == cids[i]] = cids[0]
        
        self.need_maintain_key_frames = True
            
            
            
        
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
        """Plot syllables bar timeline. Delegates to castle.visualization."""
        if self.need_maintain_key_frames:
            self.maintain_key_frames()

        from castle.visualization.embedding_plots import plot_syllables_bar as _plot_bar
        _plot_bar(self.syllables, self.key_frames, self.meta, 
                  palette_fn=self.palette, legend=legend)


def gen_palette(avoid):
    res = [it for it in _palette if not it in avoid]
    if len(res) == 0:
        return _palette
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
        """Plot embedding scatter. Delegates to castle.visualization."""
        assert hasattr(self, 'embedding')
        from castle.visualization.embedding_plots import plot_focus_embedding as _plot_focus
        cluster = self.cluster if hasattr(self, 'cluster') else None
        _plot_focus(self.embedding, self.focus, cluster=cluster, 
                    palette_fn=self.palette, dims=dimensions, legend=legend)