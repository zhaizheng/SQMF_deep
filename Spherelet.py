import numpy as np

def spherelet(DATA_TOBE, Data, k, d):
    """
    Perform Spherelet transformation on input data.
    
    Parameters:
    - DATA_TOBE: Input data to transform (m, N)
    - Data: Reference data (m, M)
    - k: Number of neighbors to use for each point
    - d: Dimensionality of the subspace (for SVD)
    
    Returns:
    - Output: The transformed data (m, N)
    - radius: Radius of each spherelet
    - C: Centers of each spherelet
    - ind: Indices of neighbors for each point
    """
    Output = np.zeros_like(DATA_TOBE)
    C = np.zeros_like(DATA_TOBE)
    radius = np.zeros(DATA_TOBE.shape[1])

    for i in range(Output.shape[1]):
        # Sorting to find nearest neighbors (excluding the point itself)
        distances = np.sum((Data - DATA_TOBE[:, i:i+1]) ** 2, axis=0)
        ind = np.argsort(distances)
        S = Data[:, ind[1:k+1]]  # Select k neighbors
        c1 = np.mean(S, axis=1)

        # Singular Value Decomposition (SVD)
        U, _, _ = np.linalg.svd(S - c1[:, np.newaxis], full_matrices=False)
        Y = c1[:, np.newaxis] + U @ U.T @ (S - c1[:, np.newaxis])
        
        # Compute spherelet center and radius
        c, r = compute_spherelet_center(Y)
        
        # Project data point onto spherelet subspace
        PU = U @ U.T @ (DATA_TOBE[:, i:i+1] - c)
        res = c + r * PU / (np.linalg.norm(PU) + np.finfo(float).eps)

        if np.isnan(res[0]):
            raise ValueError("Result is NaN, check the calculations.")

        # Store results
        Output[:, i] = res.flatten()
        radius[i] = r
        C[:, i] = c.flatten()

    return Output, radius, C, ind


def compute_spherelet_center(Y):
    """
    Compute the center and radius of the spherelet.
    
    Parameters:
    - Y: Data points (m, k) for spherelet
    
    Returns:
    - c: Center of the spherelet (m,)
    - r: Radius of the spherelet
    """
    center = np.mean(Y, axis=1)
    H = (Y - center[:, np.newaxis]) @ (Y - center[:, np.newaxis]).T
    center2 = np.mean(np.diag(Y.T @ Y))
    xi = np.zeros(Y.shape[0])

    # Compute xi
    for j in range(Y.shape[1]):
        xi += (Y[:, j].T @ Y[:, j] - center2) * (Y[:, j] - center)

    # Calculate center c using pseudoinverse
    c = np.linalg.pinv(H) @ xi / 2

    # Compute radius r
    r = 0
    for j in range(Y.shape[1]):
        r += np.linalg.norm(Y[:, j] - c) ** 2 / Y.shape[1]
    
    r = np.sqrt(r)

    if np.isnan(r):
        raise ValueError("Radius is NaN, check the calculations.")
    
    return c, r
