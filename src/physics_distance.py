import numpy as np


def physics_distance_matrix(X, feature_names, lambda_phys=1.0):
    """
    Compute physics-aware pairwise distance matrix.
    Combines Euclidean + astrophysical penalties.
    """

    n = X.shape[0]
    dist = np.zeros((n, n))

    # indices for key physical features
    def idx(name):
        return feature_names.index(name) if name in feature_names else None

    i_flux = idx("P_FLUX")
    i_temp = idx("P_TEMP_EQUIL")
    i_radius = idx("P_RADIUS")
    i_mass = idx("P_MASS")

    for i in range(n):
        for j in range(i, n):

            # base Euclidean distance
            d_base = np.linalg.norm(X[i] - X[j])

            d_phys = 0.0

            # flux consistency penalty
            if i_flux is not None:
                d_phys += abs(X[i, i_flux] - X[j, i_flux])

            # temperature consistency penalty
            if i_temp is not None:
                d_phys += abs(X[i, i_temp] - X[j, i_temp])

            # radius-mass coupling penalty (physics coupling)
            if i_radius is not None and i_mass is not None:
                ratio_i = X[i, i_radius] / (X[i, i_mass] + 1e-9)
                ratio_j = X[j, i_radius] / (X[j, i_mass] + 1e-9)
                d_phys += abs(ratio_i - ratio_j)

            d_total = d_base + lambda_phys * d_phys

            dist[i, j] = d_total
            dist[j, i] = d_total

    return dist
