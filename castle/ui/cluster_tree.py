"""
castle/ui/cluster_tree.py
Tree view rendering for cluster hierarchy.
"""

from collections import Counter


def _cumulative_counts(cluster_meta: dict, direct_counts: Counter) -> dict:
    """Compute per-node frame counts that include all descendant frames.

    Hierarchy is encoded with '/' in the cluster name
    (e.g. 'root' → 'root/a0' → 'root/a0/b1').

    Propagation runs bottom-up: each node adds its own direct count to
    its parent, so every ancestor accumulates the full subtree total.

    Args:
        cluster_meta: {id: {name, color, ...}}
        direct_counts: Counter({id: direct_frame_count})

    Returns:
        dict {id: cumulative_count}
    """
    # Start from direct counts
    cum = {cid: direct_counts.get(cid, 0) for cid in cluster_meta}

    # Build name → id lookup
    name_to_id = {meta['name']: cid for cid, meta in cluster_meta.items()}

    # Process deepest nodes first (bottom-up)
    sorted_items = sorted(
        cluster_meta.items(),
        key=lambda x: -len(x[1]['name'].split('/')),
    )

    for cid, meta in sorted_items:
        name = meta['name']
        parts = name.split('/')
        if len(parts) <= 1:
            continue  # root node, no parent
        parent_name = '/'.join(parts[:-1])
        parent_id = name_to_id.get(parent_name)
        if parent_id is not None:
            cum[parent_id] = cum.get(parent_id, 0) + cum.get(cid, 0)

    return cum


def build_cluster_tree_markdown(cluster_meta, cluster_array):
    """Build a markdown tree from cluster metadata.

    Parses hierarchical cluster names using '/' as the hierarchy delimiter
    (e.g. root → root/a0 → root/a0/b1) and renders them as an indented tree.
    Parent nodes are always shown; their count includes all descendant frames.

    Args:
        cluster_meta: dict {id: {name, color}}
        cluster_array: numpy array of cluster assignments

    Returns:
        Markdown string with tree visualization
    """
    direct = Counter(cluster_array.tolist())
    cum = _cumulative_counts(cluster_meta, direct)

    items = sorted(cluster_meta.items(), key=lambda x: x[1]['name'])

    lines = []
    for cid, meta in items:
        name = meta['name']
        total_count = cum.get(cid, 0)

        if total_count == 0:
            continue

        parts = name.split('/')
        depth = max(len(parts) - 1, 0)
        indent = '    ' * depth

        prefix = '├── ' if depth > 0 else ''
        line = f"{indent}{prefix}🔸 **{name}** ({total_count} bins)"
        lines.append(line)

    if not lines:
        return "*No clusters yet*"

    return "### 📊 Cluster Tree\n\n" + "\n\n".join(lines)


def build_cluster_tree_choices(cluster_meta, cluster_array):
    """Build tree-formatted choices for gr.Radio.

    Returns list of (display_label, value) where display_label has tree
    formatting (indentation, branch chars) and value is the cluster name.

    Parent nodes are included and their frame count includes all descendants.
    Hierarchy is encoded with '/' as the delimiter (e.g. 'root/a0/b1').

    Args:
        cluster_meta: dict {id: {name, color}}
        cluster_array: numpy array of cluster assignments

    Returns:
        List of (label, value) tuples for gr.Radio choices
    """
    direct = Counter(cluster_array.tolist())
    cum = _cumulative_counts(cluster_meta, direct)

    items = sorted(cluster_meta.items(), key=lambda x: x[1]['name'])

    choices = []
    for cid, meta in items:
        name = meta['name']
        total_count = cum.get(cid, 0)

        if total_count == 0:
            continue

        parts = name.split('/')
        depth = len(parts) - 1
        indent = '  ' * depth

        color = meta.get('color', 'grey')
        is_leaf = not any(
            other_meta['name'].startswith(name + '/')
            for other_meta in cluster_meta.values()
        )
        # Leaf clusters: colored icon. Parent/container nodes: folder icon.
        icon = ('🟢' if color != 'grey' else '📁') if is_leaf else '📂'
        prefix = '├── ' if depth > 0 else ''

        display_label = f"{indent}{prefix}{icon} {name} ({total_count} frames)"
        choices.append((display_label, name))

    return choices
