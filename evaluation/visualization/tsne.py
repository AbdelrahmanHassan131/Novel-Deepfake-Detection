"""
t-SNE Visualization.

Generates t-SNE plots from pre-extracted feature embeddings.
Refactored from ``MyModels/plot_t_SNE.py`` into a clean, reusable module.

Usage::

    from evaluation.visualization import plot_tsne

    plot_tsne(embeddings, labels,
              category_names=['Real', 'Fake'],
              save_path='tsne.png')
"""

import os
import numpy as np

# Limit thread usage for t-SNE
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _create_tsne(perplexity=30, max_iter=1000, random_state=42, verbose=0):
    import inspect
    from sklearn.manifold import TSNE
    kwargs = {
        'n_components': 2,
        'perplexity': perplexity,
        'random_state': random_state,
        'verbose': verbose,
    }
    sig = inspect.signature(TSNE.__init__)
    if 'max_iter' in sig.parameters:
        kwargs['max_iter'] = max_iter
    elif 'n_iter' in sig.parameters:
        kwargs['n_iter'] = max_iter
    return TSNE(**kwargs)


def plot_tsne(embeddings, labels, category_names=None, save_path=None,
              show=False, title='t-SNE Visualization',
              perplexity=30, max_iter=1000, max_samples=10000,
              colors=None, figsize=(12, 10)):
    """
    Generate a t-SNE scatter plot from feature embeddings.

    Args:
        embeddings (np.ndarray): Feature embeddings, shape ``(N, D)``.
        labels (np.ndarray): Integer category labels, shape ``(N,)``.
        category_names (list[str], optional): Names for each category.
        save_path (str, optional): File path to save the plot.
        show (bool): If True, display the plot interactively.
        title (str): Plot title.
        perplexity (float): t-SNE perplexity parameter.
        max_iter (int): Maximum iterations for t-SNE.
        max_samples (int): Subsample if dataset exceeds this.
        colors (list[str], optional): Colour for each category.
        figsize (tuple): Figure size.

    Returns:
        str or None: Path to the saved plot.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels).ravel()

    # Subsample if too large
    if len(labels) > max_samples:
        idx = np.random.choice(len(labels), max_samples, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        print(f'[t-SNE] Subsampled to {max_samples} samples')

    # Determine categories
    unique_labels = np.unique(labels)
    if category_names is None:
        category_names = [f'Class {i}' for i in unique_labels]

    if colors is None:
        default_colors = [
            '#E63946', '#2A9D8F', '#E76F51', '#457B9D', '#F4A261',
            '#9B5DE5', '#00BBF9', '#00F5D4', '#F15BB5', '#D4A373',
            '#2EC4B6', '#C1121F', '#3A0CA3', '#4CC9F0', '#FB5607',
            '#8338EC', '#FF006E', '#38B000', '#FFBE0B', '#5E60CE'
        ]
        if len(unique_labels) <= len(default_colors):
            colors = default_colors[:len(unique_labels)]
        else:
            cmap = plt.get_cmap('tab20' if len(unique_labels) <= 20 else 'nipy_spectral')
            colors = [cmap(i / max(1, len(unique_labels) - 1)) for i in range(len(unique_labels))]

    # Compute t-SNE
    print(f'[t-SNE] Computing with perplexity={perplexity}, '
          f'max_iter={max_iter}...')
    tsne = _create_tsne(perplexity=perplexity, max_iter=max_iter, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    for idx, label_val in enumerate(unique_labels):
        mask = labels == label_val
        name = (category_names[idx]
                if idx < len(category_names)
                else f'Class {label_val}')
        color = colors[idx % len(colors)]
        ax.scatter(
            embeddings_2d[mask, 0], embeddings_2d[mask, 1],
            c=color, label=name, alpha=0.6, s=50,
            edgecolors='w', linewidth=0.5,
        )

    if len(unique_labels) > 5:
        ax.legend(fontsize=10, markerscale=1.3, loc='center left', bbox_to_anchor=(1.02, 0.5))
    else:
        ax.legend(fontsize=12, markerscale=1.5, loc='best')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    result_path = None
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        result_path = save_path
        print(f'[t-SNE] Saved: {save_path}')

    if show:
        plt.show()
    plt.close(fig)

    return result_path


def plot_tsne_comparison(embeddings_dict, labels, category_names=None,
                         save_path=None, show=False, perplexity=30,
                         max_iter=1000, colors=None):
    """
    Create a side-by-side t-SNE comparison of multiple embedding sets.

    Args:
        embeddings_dict (dict): Mapping ``{name: embeddings_array}``.
        labels (np.ndarray): Shared labels for all embedding sets.
        category_names (list[str], optional): Category names.
        save_path (str, optional): File path to save the combined plot.
        show (bool): If True, display interactively.
        perplexity (float): t-SNE perplexity.
        max_iter (int): t-SNE max iterations.
        colors (list[str], optional): Colours per category.

    Returns:
        str or None: Path to the saved plot.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    labels = np.asarray(labels).ravel()
    unique_labels = np.unique(labels)
    if category_names is None:
        category_names = [f'Class {i}' for i in unique_labels]
    if colors is None:
        default_colors = [
            '#E63946', '#2A9D8F', '#E76F51', '#457B9D', '#F4A261',
            '#9B5DE5', '#00BBF9', '#00F5D4', '#F15BB5', '#D4A373',
            '#2EC4B6', '#C1121F', '#3A0CA3', '#4CC9F0', '#FB5607',
            '#8338EC', '#FF006E', '#38B000', '#FFBE0B', '#5E60CE'
        ]
        colors = default_colors[:len(unique_labels)]

    n_plots = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, embeddings) in zip(axes, embeddings_dict.items()):
        print(f'[t-SNE] Computing for {name}...')
        tsne = _create_tsne(
            perplexity=perplexity, max_iter=max_iter,
            random_state=42, verbose=0,
        )
        e2d = tsne.fit_transform(np.asarray(embeddings))

        for idx, label_val in enumerate(unique_labels):
            mask = labels == label_val
            cat_name = (category_names[idx]
                        if idx < len(category_names)
                        else f'Class {label_val}')
            ax.scatter(
                e2d[mask, 0], e2d[mask, 1],
                c=colors[idx % len(colors)], label=cat_name,
                alpha=0.6, s=30, edgecolors='w', linewidth=0.5,
            )

        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE Component 1', fontsize=11)
        ax.set_ylabel('t-SNE Component 2', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, markerscale=1.2)

    fig.suptitle('t-SNE Embedding Comparison',
                 fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()

    result_path = None
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        result_path = save_path
        print(f'[t-SNE] Saved comparison: {save_path}')

    if show:
        plt.show()
    plt.close(fig)

    return result_path
