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

        # Skip the init cluster
        if name == 'init':
            continue

        # Determine depth from name parts
        parts = name.split('_')
        depth = len(parts) - 1  # 'root' = 0, 'root_a0' = 1, etc.
        indent = '\u00a0\u00a0\u00a0\u00a0' * depth  # Non-breaking spaces for indent

        _color = meta.get('color', 'grey')
        prefix = '├── ' if depth > 0 else ''
        line = f"{indent}{prefix}🔸 **{name}** ({count} bins)"
        lines.append(line)

    if not lines:
        return "*No clusters yet*"

    return "### 📊 Cluster Tree\n\n" + "\n\n".join(lines)
