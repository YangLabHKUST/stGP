import numpy as np

def standardize_coords_list(coords_list):
    out = []
    for coords in coords_list:
        coords = np.asarray(coords, dtype=float)
        mu  = coords.mean(axis=0, keepdims=True)
        std = coords.std(axis=0, ddof=1, keepdims=True)
        out.append((coords - mu) / std)
    return out

def standardize_ages(ages):
    ages = np.asarray(ages, dtype=float).ravel()
    if ages.size < 2:
        return np.zeros_like(ages)
    std = ages.std(ddof=1)
    return (ages - ages.mean()) / std

def stack_blocks(blocks):
    """Stack a list of 2D arrays into a single (N, G) array."""
    if isinstance(blocks, np.ndarray):
        if blocks.ndim != 2:
            raise ValueError("Expected 2D array.")
        return blocks
    return np.vstack(blocks)


def log1p_counts(Y_list):
    """Apply log1p to a list of count matrices."""
    return [np.log1p(np.asarray(Y, dtype=float)) for Y in Y_list]


def demean_genes(Y_list):
    """
    Demean genes across all cells, returning (demeaned_list, gene_means).
    """
    Y_stack = stack_blocks(Y_list).astype(float)
    gene_means = Y_stack.mean(axis=0, keepdims=True)
    out = []
    off = 0
    for Y in Y_list:
        n = Y.shape[0]
        out.append(np.asarray(Y, dtype=float) - gene_means)
        off += n
    return out, gene_means.ravel()


def log1p_demean(Y_list):
    """
    Apply log1p then per-gene demeaning across all cells.
    """
    log_list = log1p_counts(Y_list)
    return demean_genes(log_list)


def library_normalize(Y_list, target_sum=1e4, eps=1e-12):
    """
    Normalize each cell to a target library size.
    """
    out = []
    for Y in Y_list:
        Y = np.asarray(Y, dtype=float)
        lib = Y.sum(axis=1, keepdims=True)
        scale = target_sum / np.maximum(lib, eps)
        out.append(Y * scale)
    return out


def log1p_normalize(Y_list, target_sum=250.0, eps=1e-12):
    """
    Normalize counts per cell to target_sum, then apply log1p.
    """
    normed = library_normalize(Y_list, target_sum=target_sum, eps=eps)
    return [np.log1p(np.asarray(Y, dtype=float)) for Y in normed]


def log1p_norm_list(Y_list, target_sum=250.0, eps=1e-12):
    """
    Alias for the normalized log1p view (normalize_total -> log1p).
    """
    return log1p_normalize(Y_list, target_sum=target_sum, eps=eps)


def log1p_norm_centered_list(Y_list, target_sum=250.0, eps=1e-12):
    """
    Normalize counts per cell to target_sum, log1p, then per-gene center.
    """
    log_list = log1p_normalize(Y_list, target_sum=target_sum, eps=eps)
    return demean_genes(log_list)
