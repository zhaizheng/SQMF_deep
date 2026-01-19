import numpy as np

def linear_mfit(Data, points, d, k):
    """
    Perform linear manifold fitting (MFit) on input points using kernel methods.
    
    Parameters:
    - Data: The dataset (m, N)
    - points: Points to transform (m, M)
    - d: Dimensionality of the subspace
    - k: Number of nearest neighbors
    
    Returns:
    - new_points_linear: Transformed points (m, M)
    - ZT: List of transformation matrices for each point
    """
    new_points_linear = np.zeros_like(points)
    ZT = [None] * points.shape[1]
    
    for i in range(points.shape[1]):
        data_old = np.zeros_like(points[:, i])
        data_move = points[:, i]
        sk = 1
        
        while np.linalg.norm(data_old - data_move) > 1e-7 and sk < 50:
            data_old = data_move
            r = find_sigma(data_move, Data, k)
            direction, P = mfit(data_old, Data, r, beta=0.1, dim=d, step=0.5)
            data_move = data_old + 0.5 * direction
            sk += 1
        
        new_points_linear[:, i] = data_move
        ZT[i] = P
    
    return new_points_linear, ZT


def find_sigma(x, Data, k):
    """
    Find the sigma value (the radius of the kernel) based on nearest neighbors.
    
    Parameters:
    - x: Current point (m, 1)
    - Data: The dataset (m, N)
    - k: Number of nearest neighbors
    
    Returns:
    - sigma: The kernel bandwidth (scalar)
    """
    s_distance = np.sum((Data - x[:, np.newaxis])**2, axis=0)
    ind = np.argsort(s_distance)
    Neig = Data[:, ind[1:k+1]]
    sigma = np.max(np.sqrt(np.sum((Neig - x[:, np.newaxis])**2, axis=0)))
    return sigma


def mfit(x, data, r, beta, dim, step):
    """
    Perform manifold fitting by computing the direction of movement for each point.
    
    Parameters:
    - x: Current point (m, 1)
    - data: Dataset (m, N)
    - r: Kernel radius (sigma)
    - beta: Kernel exponent
    - dim: Dimensionality of the subspace
    - step: Step size for updating the point
    
    Returns:
    - direction: The movement direction (m, 1)
    - P: The transformation matrix
    """
    d = data.shape[0]
    n = data.shape[1]
    d2 = 1 - np.sum((data - x[:, np.newaxis])**2, axis=0) / (r**2)
    d2[d2 < 0] = 0
    alpha_tilde = d2**beta
    alpha_sum = np.sum(alpha_tilde)
    alpha = alpha_tilde / alpha_sum
    
    Ns = np.zeros((d, d))
    c_vec = np.zeros(d)
    
    for i in range(n):
        if d2[i] > 0:
            ns = normal_space(data, i, r, dim)
            Ns += ns * alpha[i]
            c_vec += alpha[i] * ns @ (data[:, i] - x)
    
    U, _, _ = np.linalg.svd(Ns)
    P = U[:, :d-dim] @ U[:, :d-dim].T
    direction = step * P @ c_vec
    
    return direction, P


def normal_space(data, i, r, dim):
    """
    Compute the normal space for a point relative to the dataset.
    
    Parameters:
    - data: Dataset (m, N)
    - i: Index of the current point
    - r: Kernel radius (sigma)
    - dim: Dimensionality of the subspace
    
    Returns:
    - P: The transformation matrix (m, m)
    """
    ds = np.sum((data - data[:, i:i+1])**2, axis=0)
    ds[ds > r**2] = 0
    indicator = (ds > 0).astype(float)
    select = (data - data[:, i:i+1]) * indicator
    cor = select @ select.T
    U, _, _ = np.linalg.svd(cor)
    P = np.eye(data.shape[0]) - U[:, :dim] @ U[:, :dim].T
    return P


def shift_mean(x, Data, h, k):
    """
    Shift the mean of the points using the nearest neighbors.
    
    Parameters:
    - x: The current point (m, 1)
    - Data: The dataset (m, N)
    - h: The bandwidth for the kernel
    - k: Number of nearest neighbors
    
    Returns:
    - mean: The shifted mean (m, 1)
    """
    s_distance = np.sum((Data - x[:, np.newaxis])**2, axis=0)
    ind = np.argsort(s_distance)
    mean = np.zeros_like(x)
    s_weight = 0
    
    for i in range(k):
        w = np.exp(-np.linalg.norm(Data[:, ind[i]] - x)**2 / (h**2))
        s_weight += w
        mean += w * Data[:, ind[i]]
    
    mean /= s_weight
    return mean
