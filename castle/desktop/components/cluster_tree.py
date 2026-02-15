"""
CASTLE Desktop - Cluster Tree Widget

Displays the hierarchical cluster structure.
Supports click-to-select and right-click context menu for label/rename.
"""

from PyQt6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QMenu, QInputDialog
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor, QBrush, QAction


class ClusterTreeWidget(QTreeWidget):
    """Tree widget displaying cluster hierarchy."""

    cluster_selected = pyqtSignal(str)  # Emits cluster name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Cluster", "Count", "Color"])
        self.setColumnWidth(0, 150)
        self.setColumnWidth(1, 60)
        self.setColumnWidth(2, 50)
        self.setAlternatingRowColors(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
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
            item.setData(0, Qt.ItemDataRole.UserRole, cluster_id)

            if color and color != 'grey':
                try:
                    qcolor = QColor(color)
                    item.setBackground(2, QBrush(qcolor))
                except Exception:
                    pass

            self.addTopLevelItem(item)

        self.expandAll()

    def set_cluster_names(self, names: list):
        """Simple mode: set from a list of cluster names."""
        self.clear()
        for name in names:
            item = QTreeWidgetItem([name, "", ""])
            self.addTopLevelItem(item)

    def selected_cluster_name(self) -> str:
        """Get the name of the currently selected cluster."""
        items = self.selectedItems()
        if items:
            return items[0].text(0)
        return ""

    def _on_item_clicked(self, item, column):
        name = item.text(0)
        self.cluster_selected.emit(name)

    def _show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        menu = QMenu(self)
        rename_action = QAction("Rename Cluster...", self)
        rename_action.triggered.connect(lambda: self._rename_cluster(item))
        menu.addAction(rename_action)
        menu.exec(self.mapToGlobal(pos))

    def _rename_cluster(self, item):
        old_name = item.text(0)
        new_name, ok = QInputDialog.getText(
            self, "Rename Cluster",
            f"New name for '{old_name}':", text=old_name,
        )
        if ok and new_name.strip():
            item.setText(0, new_name.strip())
