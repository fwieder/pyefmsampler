#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 10:54:01 2025

@author: frederik
"""


import numpy as np
import matplotlib.pyplot as plt
import umap

from collections import Counter

from pyefmsampler.helpers import supports_to_binary_matrix,supp
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

def plot_reaction_frequencies(efms, model, title=None):
    """

    Parameters
    ----------
    efms : np.array
        Set of EFMs
    model : FluxCone model
    title : str, optional
        Custom plot title.

    Returns
    -------
    Plots the frequency of each reaction in the input-efms as a bar diagram.

    """
    
    rea_freqs = sum([list(supp(efm)) for efm in efms], [])
    data = sorted(Counter(rea_freqs).items())
    print(f"{len(data)} of {np.shape(model.stoich)[1]} reactions occur in the sample.")
    
    ax = plt.figure().gca()      # Get the current axes
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(model.id + " - pyefmsampler found EFMs: " + str(len(efms)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Reaction Index")
    ax.set_ylabel("Number of EFMs")
    if data:
        x_vals, y_vals = zip(*data)
        plt.bar(x_vals, y_vals)
    plt.savefig(model.id + "pyefmsampler_" + model.id +".pdf",dpi=300)
    plt.show()
    

def plot_efm_lengths(efms,model):
    """

    Parameters
    ----------
    efms : np.array
        Set of EFMs
    model : FluxCone model

    Returns
    -------
    Plots the lengths of the supports of the input-efms as a bar diagram.

    """
    
    efm_lens = [len(supp(efm)) for efm in efms]
    
    data = sorted(Counter(efm_lens).items())
   
    
    ax = plt.figure().gca()      # Get the current axes
    ax.set_title(model.id + " - pyefmsampler found EFMs: " + str(len(efms)))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("EFM_length")
    ax.set_ylabel("Number of EFMs")
    plt.bar(list(zip(*data))[0],list(zip(*data))[1])
    plt.savefig(model.id + "pyefmsampler_" + model.id +".pdf",dpi=300)
    plt.show()

def umap_efms(efms, labels=None,neighbors = 10):
    efms = supports_to_binary_matrix([supp(efm) for efm in efms], len(efms[0]))

    # Fit UMAP
    umap_model = umap.UMAP(n_components=2, metric='hamming',n_neighbors=neighbors,min_dist=0.1)
    embedding = umap_model.fit_transform(efms)
    # Compute trustworthiness score
    trust = trustworthiness(efms, embedding, n_neighbors=neighbors)


    # Optional: Silhouette score (requires labels)
    if labels is not None:
        sil_score = silhouette_score(embedding, labels)
        print(f"Silhouette Score (0-1): {sil_score:.3f}")

    # Pairwise distance variance (spread)
    pairwise_dist = pairwise_distances(embedding)
    spread = np.std(pairwise_dist)
    print(f"Pairwise Distance Std Dev (Spread): {spread:.3f}")

    # Create a color gradient based on entry order
    n = efms.shape[0]
    colors = np.linspace(0, 1, n)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap='Reds',
        alpha=0.8,
        s=5
    )

    plt.title("UMAP Projection (Hamming Distance) - " + str(len(efms)) + "EFMs")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("EFM Index (Early → Late)")
    
    plt.figtext(0.5, -0.05, f"Trustworthiness (0-1): {trust:.3f}" + "    " + f"Pairwise Distance Std Dev (Spread): {spread:.3f}" + "    " + f"Nearest neighbors: {neighbors}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    return embedding


def pca_efms(efms, labels=None, neighbors=10, random_state=42):
    """Perform PCA on EFMs after scaling each vector to unit length."""
    all_efms = [np.asarray(efm, dtype=float) for efm in efms]
    try:
        X = np.stack(all_efms)
    except ValueError as e:
        raise ValueError("All EFMs must have the same dimensionality for PCA.") from e

    norms = np.linalg.norm(X, axis=1)
    nonzero = norms != 0
    if np.any(nonzero):
        X[nonzero] = X[nonzero] / norms[nonzero, np.newaxis]

    pca_model = PCA(n_components=2, random_state=random_state)
    embedding = pca_model.fit_transform(X)

    trust = trustworthiness(X, embedding, n_neighbors=neighbors)
    print(f"Trustworthiness (0-1): {trust:.3f}")

    if labels is not None:
        sil_score = silhouette_score(embedding, labels)
        print(f"Silhouette Score (0-1): {sil_score:.3f}")

    pairwise_dist = pairwise_distances(embedding)
    spread = np.std(pairwise_dist)
    print(f"Pairwise Distance Std Dev (Spread): {spread:.3f}")

    explained = np.sum(pca_model.explained_variance_ratio_)
    print(f"Explained Variance (2 components): {explained:.3f}")

    n = X.shape[0]
    colors = np.linspace(0, 1, n)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap='Reds',
        alpha=0.8,
        s=5
    )

    plt.title("PCA Projection (Scaled EFMs) - " + str(len(efms)) + " EFMs")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("EFM Index (Early → Late)")
    plt.figtext(0.5, -0.05, f"Trustworthiness: {trust:.3f}" + "    " + f"Pairwise Distance Std Dev (Spread): {spread:.3f}" + "    " + f"Explained Var: {explained:.3f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    return embedding

def umap_efm_sets(efm_sets, neighbors=10, sample_names=None, random_state=42):
    """
    efm_sets: list of lists of EFMs, e.g. [efms1, efms2, efms3]
    sample_names: optional list of names for legend
    """

    # -----------------------------
    # Flatten EFMs + track origin
    # -----------------------------
    all_efms = []
    labels = []

    for i, efms in enumerate(efm_sets):
        for efm in efms:
            all_efms.append(efm)
            labels.append(i)

    # -----------------------------
    # Convert to binary support matrix
    # -----------------------------
    supports = [supp(efm) for efm in all_efms]
    n_reactions = len(all_efms[0])
    X = supports_to_binary_matrix(supports, n_reactions)

    # -----------------------------
    # UMAP embedding
    # -----------------------------
    umap_model = umap.UMAP(
        n_components=2,
        metric='hamming',
        n_neighbors=neighbors,
        min_dist=0.1,
        random_state=random_state
    )

    embedding = umap_model.fit_transform(X)

    # -----------------------------
    # Quality metrics
    # -----------------------------
    trust = trustworthiness(X, embedding, n_neighbors=neighbors)
    print(f"Trustworthiness (0-1): {trust:.3f}")

    pairwise_dist = pairwise_distances(embedding)
    spread = np.std(pairwise_dist)
    print(f"Pairwise Distance Std Dev (Spread): {spread:.3f}")

    # -----------------------------
    # Plot (color by sample)
    # -----------------------------
    plt.figure(figsize=(10, 7))

    labels = np.array(labels)
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        idx = labels == lab
        name = sample_names[lab] if sample_names else f"Sample {lab+1}"

        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            label=name,
            alpha=0.7,
            s=8
        )

    plt.title(f"UMAP Projection (Hamming) – {len(all_efms)} EFMs")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend()

    plt.figtext(
        0.5,
        -0.05,
        f"Trustworthiness: {trust:.3f}    Spread: {spread:.3f}    Neighbors: {neighbors}",
        ha="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

    return embedding, labels


def pca_efm_sets(efm_sets, neighbors=10, sample_names=None, random_state=42):
    """
    efm_sets: list of lists of EFMs, e.g. [efms1, efms2, efms3]
    sample_names: optional list of names for legend
    """

    # -----------------------------
    # Flatten EFMs + track origin
    # -----------------------------
    all_efms = []
    labels = []

    for i, efms in enumerate(efm_sets):
        for efm in efms:
            all_efms.append(np.asarray(efm, dtype=float))
            labels.append(i)

    # -----------------------------
    # Convert to data matrix and normalize lengths
    # -----------------------------
    try:
        X = np.stack(all_efms)
    except ValueError as e:
        raise ValueError("All EFMs must have the same dimensionality for PCA.") from e

    norms = np.linalg.norm(X, axis=1)
    nonzero = norms != 0
    if np.any(nonzero):
        X[nonzero] = X[nonzero] / norms[nonzero, np.newaxis]

    # -----------------------------
    # PCA embedding
    # -----------------------------
    pca_model = PCA(n_components=2, random_state=random_state)
    embedding = pca_model.fit_transform(X)

    # -----------------------------
    # Quality metrics
    # -----------------------------
    trust = trustworthiness(X, embedding, n_neighbors=neighbors)
    print(f"Trustworthiness (0-1): {trust:.3f}")

    pairwise_dist = pairwise_distances(embedding)
    spread = np.std(pairwise_dist)
    print(f"Pairwise Distance Std Dev (Spread): {spread:.3f}")

    explained = np.sum(pca_model.explained_variance_ratio_)
    print(f"Explained Variance (2 components): {explained:.3f}")

    # -----------------------------
    # Plot (color by sample)
    # -----------------------------
    plt.figure(figsize=(10, 7))

    labels = np.array(labels)
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        idx = labels == lab
        name = sample_names[lab] if sample_names else f"Sample {lab+1}"

        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            label=name,
            alpha=0.7,
            s=8
        )

    plt.title(f"PCA Projection (Scaled EFMs) – {len(all_efms)} EFMs")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()

    plt.figtext(
        0.5,
        -0.05,
        f"Trustworthiness: {trust:.3f}    Spread: {spread:.3f}    Explained Var: {explained:.3f}",
        ha="center",
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

    return embedding, labels

def umap_efm_sample(sample,full_set, labels=None, neighbors=10, name = ""):
    full_set = supports_to_binary_matrix([supp(efm) for efm in full_set], len(full_set[0]))
    sample = supports_to_binary_matrix([supp(efm) for efm in sample], len(sample[0]))
    # Define a helper to convert rows to consistent tuples of ints
    def row_key(row):
        return tuple(int(x) for x in row)

    # Fit UMAP on full set
    umap_model = umap.UMAP(n_components=2, metric='hamming', n_neighbors=neighbors, min_dist=0.1,random_state=42)
    embedding_full = umap_model.fit_transform(full_set)

    # Map sample indices (requires exact row match)
    full_rows_str = {row_key(row): idx for idx, row in enumerate(full_set)}
    try:
        sample_indices = [full_rows_str[row_key(row)] for row in sample]
    except KeyError as e:
        raise ValueError("Some sample elements are not present in the full set. Please verify subset relationship.") from e

    embedding_sample = embedding_full[sample_indices]

    # Trustworthiness and spread for the sample
    trust = trustworthiness(full_set, embedding_full, n_neighbors=neighbors)
    spread = np.std(pairwise_distances(embedding_full))
    
    # Optional: Silhouette score if labels are provided
    if labels is not None:
        sil_score = silhouette_score(embedding_sample, labels)
        print(f"Silhouette Score (0-1): {sil_score:.3f}")

    # Color gradient for sample points based on appearance
    colors = np.linspace(0, 1, len(sample))

    # Plotting
    plt.figure(figsize=(10, 7))
    plt.scatter(embedding_full[:, 0], embedding_full[:, 1], c='lightgray', alpha=0.5, s=5, label='Full set')
    scatter = plt.scatter(embedding_sample[:, 0], embedding_sample[:, 1], c=colors, cmap='Reds', s=5, alpha=0.9, label='Sample subset')

    plt.title(f"UMAP Projection {name} — Sample ({len(sample)} EFMs) vs Full Set ({len(full_set)} EFMs)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Sample Index (Early → Late)")
    sample_legend = Line2D( [0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Sample subset')
    
    plt.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=6, label='Full set'), sample_legend], loc='upper right')
    
    plt.figtext(0.5, -0.05, f"Trustworthiness: {trust:.3f}    Spread: {spread:.3f}    n_neighbors: {neighbors}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    return embedding_full, embedding_sample
