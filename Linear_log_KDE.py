import numpy as np

def linear_log_KDE(k, Data, points, d):
    """
    Perform linear log KDE transformation.
    
    Parameters:
    - k: Number of nearest neighbors
    - Data: The dataset (m, N)
    - points: Points to transform (m, M)
    - d: Dimensionality of the subspace (for fitting)
    
    Returns:
    - new_points_linear: The transformed points (m, M)
    - ZT: List of transformation matrices for each point
    """
    D = Data.shape[0]
    new_points_linear = np.zeros_like(points)
    ZT = [None] * points.shape[1]
    
    for i in range(points.shape[1]):
        data_old = np.zeros_like(points[:, i])
        data_move = points[:, i]
        sk = 1
        
        while np.linalg.norm(data_old - data_move) > 1e-7 and sk < 50:
            data_old = data_move
            sigma = find_sigma(data_move, Data, k)
            x = shift_mean(data_move, Data, sigma, k)
            _, _, _, _, U_v = fitting(x, Data, sigma, D, d)
            data_move = data_old + U_v @ U_v.T @ (x - data_old)
            sk += 1
        
        ZT[i] = U_v @ U_v.T
        new_points_linear[:, i] = data_move
    
    return new_points_linear, ZT


def find_sigma(x, Data, k):
    """
    Find the sigma value (the radius of the kernel) based on nearest neighbors.
    
    Parameters:
    - x: The current point (m, 1)
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


def fitting(x, Data, sigma, D, d):
    """
    Perform fitting based on the coordinates and weights.
    
    Parameters:
    - x: The current point (m, 1)
    - Data: The dataset (m, N)
    - sigma: The kernel bandwidth
    - D: The dimensionality of the data
    - d: The dimensionality of the subspace
    
    Returns:
    - Theta: The fitted transformation matrix
    - Co_h, Co_v: The coordinate transformations
    - U_h, U_v: The subspaces for horizontal and vertical directions
    """
    Co_h, Co_v, U_h, U_v = coordinate(x, Data, sigma, D, d)
    W = build_W(x, Data, sigma)
    _, Theta = least_square(Co_h, Co_v, W)
    return Theta, Co_h, Co_v, U_h, U_v


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


def build_W(x, A, h):
    """
    Build the weight matrix for the kernel.
    
    Parameters:
    - x: The current point (m, 1)
    - A: The dataset (m, N)
    - h: The bandwidth for the kernel
    
    Returns:
    - W: The weight matrix (N, N)
    """
    centered = A - x[:, np.newaxis]
    W = np.zeros(A.shape[1])
    
    for k in range(A.shape[1]):
        W[k] = np.exp(-np.linalg.norm(centered[:, k])**2 / (h**2))
    
    return np.diag(W)


def least_square(Tau, Co_v, W):
    """
    Perform least squares fitting for the coordinates and weights.
    
    Parameters:
    - Tau: The transformed coordinates (d, N)
    - Co_v: The vertical coordinates (d, N)
    - W: The weight matrix (N, N)
    
    Returns:
    - theta: The fitting parameters
    - Theta: The transformation matrices
    """
    d = Tau.shape[0]
    ind = np.triu_indices(d)
    d2 = d * (d + 1) // 2
    Theta = [None] * Co_v.shape[0]
    
    for j in range(Co_v.shape[0]):
        G = np.zeros((d2, Tau.shape[1]))
        
        for i in range(Tau.shape[1]):
            A = np.outer(Tau[:, i], Tau[:, i])
            G[:, i] = A[ind]
        
        theta = np.linalg.solve(G @ W @ G.T, G @ W @ Co_v[j, :])
        Theta_temp = np.zeros((d, d))
        Theta_temp[ind] = theta / 2
        Theta[j] = Theta_temp + Theta_temp.T
    
    return theta, Theta


def coordinate(x, A, h, D, d):
    """
    Compute the coordinates of the current point relative to the dataset.
    
    Parameters:
    - x: The current point (m, 1)
    - A: The dataset (m, N)
    - h: The bandwidth for the kernel
    - D: The dimensionality of the dataset
    - d: The dimensionality of the subspace
    
    Returns:
    - Co_h: The horizontal coordinates (d, N)
    - Co_v: The vertical coordinates (d, N)
    - U_h: The horizontal subspace (d, d)
    - U_v: The vertical subspace (d, D-d)
    """
    C = np.zeros((x.shape[0], x.shape[0]))
    
    for i in range(A.shape[1]):
        a = x - A[:, i]
        C += np.exp(-np.linalg.norm(a)**2 / (h**2)) * np.outer(a, a)
    
    U, _, _ = np.linalg.svd(C)
    U_h = U[:, :d]
    U_v = U[:, d:D]
    Co_h = U_h.T @ (A - x[:, np.newaxis])
    Co_v = U_v.T @ (A - x[:, np.newaxis])
    
    return Co_h, Co_v, U_h, U_v
