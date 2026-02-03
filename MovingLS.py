import numpy as np

def moving_ls(Data, inputs, neig, d, max_iter=300, tol=1e-4):
    """
    Perform a moving least squares (MLS) fitting algorithm.
    
    Parameters:
    - Data: (m, N) input data matrix where each column is a data point.
    - inputs: (m, M) input locations for MLS evaluation.
    - neig: Number of nearest neighbors for each input.
    - d: Dimensionality of the subspace.
    - max_iter: Maximum iterations for the optimization (default 300).
    - tol: Tolerance for convergence (default 1e-4).
    
    Returns:
    - Result: The fitted values (m, M) for each input.
    - T_list: List of transformation matrices for each input.
    - re: Dictionary containing additional results (Q, X, U, T).
    """
    Result = []
    Q_all = []
    T_list = [None] * inputs.shape[1]
    
    for i in range(inputs.shape[1]):
        x = inputs[:, i:i+1]  # Current input point (m, 1)

        # Find sigma (radius) and neighbors
        h, Datap, _ = find_sigma(x, Data, neig, int(np.floor(neig)))
        
        # Find the origin and the corresponding subspace U
        q, _, _, U = find_origin(x, Datap, h, d, max_iter, tol)
        
        # Perform least fitting
        result = least_fitting(Datap, U, q, h)
        
        # Store results
        Result.append(result)
        Q_all.append(q)
        T_list[i] = U @ U.T

    # Pack results into dictionary
    re = {
        "Q": np.vstack(Q_all),
        "X": np.vstack(Result),
        "U": U,
        "T": T_list
    }

    return np.vstack(Result), T_list, re


def find_origin(x, Data, h, d, max_iter=300, tol=1e-4):
    """
    Iteratively refine the origin q and subspace U using principal component analysis (PCA).
    
    Parameters:
    - x: The current input point (m, 1).
    - Data: The input data (m, N).
    - h: The bandwidth parameter for weighting.
    - d: Dimensionality of the subspace.
    - max_iter: Maximum iterations for optimization.
    - tol: Tolerance for convergence.
    
    Returns:
    - q: The refined origin.
    - k: The number of iterations performed.
    - Q_hist: History of q values during iterations.
    - C: The principal components corresponding to the subspace.
    """
    q = x.copy()  # Initialize origin as input point
    U, _center = principal(Data, h, q, d)
    k = 0
    Q_hist = []

    while k < max_iter:
        Data_c = Data - q  # Center data around the current origin
        Theta = build_theta(Data, h, q)
        R = Data_c @ Theta
        Tau = U[:, :d].T @ R
        Tau1 = np.vstack([np.ones((1, Tau.shape[1])), Tau])

        # Compute the update matrix A
        A = (R @ Tau1.T) @ np.linalg.pinv(Tau1 @ Tau1.T)
        q_temp = q + A[:, 0:1]

        # Perform QR decomposition on A[:, 1:]
        U_qr, _ = np.linalg.qr(A[:, 1:])

        # Update q based on the current subspace U
        Ud = U_qr[:, :d]
        q_new = q_temp + Ud @ (Ud.T @ (x - q_temp))

        # Check for convergence
        if np.linalg.norm(q_new - q) < tol:
            U = U_qr
            q = q_new
            break

        q = q_new
        U = U_qr
        Q_hist.append(q)
        k += 1

    C = U[:, :d]
    return q, k, Q_hist, C


def least_fitting(Data, U, q, h):
    """
    Perform least squares fitting given data and the subspace.
    
    Parameters:
    - Data: The input data (m, N).
    - U: The subspace basis (m, d).
    - q: The origin (m, 1).
    - h: The bandwidth parameter for weighting.
    
    Returns:
    - result: The fitted values (m, 1).
    """
    Tau = U.T @ (Data - q)
    T = construct_higher_order(Tau)
    Theta = build_theta(Data, h, q) ** 2
    A = (Data @ Theta @ T.T) @ np.linalg.pinv(T @ Theta @ T.T)
    return A[:, 0:1]


def construct_higher_order(Tau):
    """
    Constructs higher-order terms for fitting.
    
    Parameters:
    - Tau: The residuals (d, N).
    
    Returns:
    - T: The higher-order terms (1 + d + d(d+1)/2, N).
    """
    d, N = Tau.shape
    out_rows = 1 + d + d * (d + 1) // 2
    T = np.zeros((out_rows, N))

    # Upper-triangular indices for quadratic terms
    ind = np.triu_indices(d)

    for i in range(N):
        T[0:1+d, i] = np.concatenate(([1.0], Tau[:, i]))
        temp = np.outer(Tau[:, i], Tau[:, i])
        T[1+d:, i] = temp[ind]

    return T


def build_theta(Data, h, q):
    """
    Build the weight matrix (identity or Gaussian).
    
    Parameters:
    - Data: The input data (m, N).
    - h: The bandwidth parameter.
    - q: The origin (m, 1).
    
    Returns:
    - Theta: The weight matrix (N, N).
    """
    N = Data.shape[1]
    # Implement Gaussian weights if necessary
    return np.eye(N)


def principal(Data, h, q, d):
    """
    Perform PCA to find the principal components.
    
    Parameters:
    - Data: The input data (m, N).
    - h: The bandwidth parameter.
    - q: The origin (m, 1).
    - d: The dimensionality of the subspace.
    
    Returns:
    - U: The principal components (m, d).
    - center: The weighted center (m, 1).
    """
    Theta = build_theta(Data, h, q) ** 2
    center = (Data @ Theta).sum(axis=1, keepdims=True) / np.diag(Theta).sum()

    X = (Data - center) @ Theta @ (Data - center).T
    V, _, _ = np.linalg.svd(X, full_matrices=True)
    U = V[:, :d]
    return U, center


def find_sigma(x, Data, k, s):
    """
    Find the neighbors and the bandwidth sigma.
    
    Parameters:
    - x: The input point (m, 1).
    - Data: The input data (m, N).
    - k: The number of neighbors.
    - s: The number of points to return.
    
    Returns:
    - sigma: The bandwidth sigma.
    - Datap: The selected data points (m, s).
    - Neig: The neighboring points (m, k).
    """
    s_distance = np.sum((Data - x) ** 2, axis=0)
    ind = np.argsort(s_distance)
    Neig = Data[:, ind[1:k+1]]
    sigma = np.max(np.sqrt(np.sum((Neig - x) ** 2, axis=0)))
    t = Data.shape[1]
    Datap = Data[:, ind[1:min(t, s)]]
    return sigma, Datap, Neig
