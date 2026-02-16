"""
castle/ui/cluster_tree.py
Tree view rendering for cluster hierarchy.
"""

from collections import Counter


def build_cluster_tree_markdown(cluster_meta, cluster_array):
    """Build a markdown tree from cluster metadata.

    Parses hierarchical cluster names (root → root_a0 → root_a0_b1)
    and renders them as an indented tree with bin counts.

    Args:
        cluster_meta: dict {id: {name, color}}
        cluster_array: numpy array of cluster assignments

    Returns:
        Markdown string with tree visualization
    """
    counts = Counter(cluster_array.tolist())

    # Sort by name
    items = sorted(cluster_meta.items(), key=lambda x: x[1]['name'])

    lines = []
    for cid, meta in items:
        name = meta['name']
        count = counts.get(cid, 0)

        # Determine depth from name parts
        parts = name.split('_')
        depth = max(len(parts) - 1, 0)
        indent = '\u00a0\u00a0\u00a0\u00a0' * depth  # Non-breaking spaces for indent

        _color = meta.get('color', 'grey')
        prefix = '├── ' if depth > 0 else ''
        line = f"{indent}{prefix}🔸 **{name}** ({count} bins)"
        lines.append(line)

    if not lines:
        return "*No clusters yet*"

    return "### 📊 Cluster Tree\n\n" + "\n\n".join(lines)


def build_cluster_tree_choices(cluster_meta, cluster_array):
    """Build tree-formatted choices for gr.Radio.
    
    Returns list of (display_label, value) where display_label has tree
    formatting (indentation, branch chars) and value is the cluster name.
    
    Args:
        cluster_meta: dict {id: {name, color}}
        cluster_array: numpy array of cluster assignments
        
    Returns:
        List of (label, value) tuples for gr.Radio choices
    """
    counts = Counter(cluster_array.tolist())
    
    # Sort by name
    items = sorted(cluster_meta.items(), key=lambda x: x[1]['name'])
    
    choices = []
    for cid, meta in items:
        name = meta['name']
        count = counts.get(cid, 0)
        
        # Determine depth from name parts
        parts = name.split('_')
        depth = len(parts) - 1  # 'root' = 0, 'root_a0' = 1, etc.
        indent = '  ' * depth  # Two spaces per level
        
        color = meta.get('color', 'grey')
        # 📁 for empty (grey) clusters, 🟢 for labeled clusters
        icon = '📁' if color == 'grey' else '🟢'
        prefix = '├── ' if depth > 0 else ''
        
        # Build display label with tree formatting
        display_label = f"{indent}{prefix}{icon} {name} ({count} frames)"
        
        # Value is just the cluster name
        choices.append((display_label, name))
    
    return choices
