"""
CASTLE Desktop - Cluster Tree Widget

Displays the hierarchical cluster structure.
Allows users to see where they are in the clustering process
and select clusters for further subdivision.
"""

from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QBrush


class ClusterTreeWidget(QTreeWidget):
    """Tree widget displaying cluster hierarchy."""
    
    cluster_selected = Signal(str)  # Emits cluster name
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Cluster", "Count", "Color"])
        self.setColumnWidth(0, 150)
        self.setColumnWidth(1, 60)
        self.setColumnWidth(2, 50)
        self.setAlternatingRowColors(True)
        
        self.itemClicked.connect(self._on_item_clicked)
    
    def set_latent(self, latents):
        """Update tree from Latent object.
        
        Args:
            latents: Latent object with cluster, cluster_meta, behavior_name2cluster_id
        """
        self.clear()
        
        if latents is None:
            return
        
        cluster = latents.cluster
        cluster_meta = latents.cluster_meta
        
        for cluster_id, meta in cluster_meta.items():
            name = meta.get('name', f'C{cluster_id}')
            color = meta.get('color', 'grey')
            count = int(sum(cluster == cluster_id))
            
            item = QTreeWidgetItem([name, str(count), ""])
            
            # Set color indicator
            if color != 'grey':
                try:
                    qcolor = QColor(color)
                    item.setBackground(2, QBrush(qcolor))
                except Exception:
                    pass
            
            self.addTopLevelItem(item)
        
        self.expandAll()
    
    def selected_cluster_name(self) -> str:
        """Get the name of the currently selected cluster."""
        items = self.selectedItems()
        if items:
            return items[0].text(0)
        return ""
    
    def _on_item_clicked(self, item, column):
        """Handle tree item click."""
        name = item.text(0)
        self.cluster_selected.emit(name)
