import numpy as np
from SQMF_v3 import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_generator import *
import os

mpl.rcParams['text.usetex'] = True
os.makedirs("modified/figures", exist_ok=True)

# ---------------- Data generation ----------------

def generate_data(start, end, num):
    t = np.linspace(start, end, num)
    x = np.cos(t)
    y = np.sin(t)
    return np.vstack((x, y))

def eval_model(c, U, V, Theta, taus):
    taus = np.asarray(taus)
    out = []
    for tau in taus:
        vech_tt = vech_upper(np.outer(tau, tau))
        f = c.ravel() + U @ tau + V @ (Theta.T @ vech_tt)
        out.append(f)
    return np.column_stack(out)

def fit_and_reconstruct(XY, mode, p_value):
    Q, Theta, c, taus, err, step = quadratic_manifold_factorization(
        XY, d=1, s=1,
        eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
        T=1000, tol=1e-10, mode=mode, delta=0.1, p=p_value
    )

    taus = np.asarray(taus).reshape(-1, 1)
    U = Q[:, [0]]
    V = Q[:, [1]]

    result = eval_model(c, U, V, Theta, taus)

    t_mid = np.linspace(taus.min(), taus.max(), 50).reshape(-1, 1)
    result2 = eval_model(c, U, V, Theta, t_mid)

    return result, result2, err, step

def sample_on_sphere(n, d, seed=None):
    """Uniformly sample n points on S^{d-1}."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.T  # shape (d, n)

def k_nearest_on_sphere(X, x0, k):
    """Return k nearest neighbors of x0 from X (columns)."""
    x0 = x0 / np.linalg.norm(x0)
    dists = np.linalg.norm(X - x0, axis=0)
    idx = np.argsort(dists)[:k]
    return X[:, idx], idx, dists[idx]

# ---------------- Experiment setup ----------------

types = ['lp_p'] * 5 + ['l2'] 
labels = [
    r'$\ell_p^p, p=1.0$', r'$\ell_p^p, p=1.25$', r'$\ell_p^p, p=1.5$',
    r'$\ell_p^p, p=1.75$', r'$\ell_p^p, p=2.0$', r'$\ell_2$'
]
P_values = [1, 1.25, 1.5, 1.75, 2, 2]

partition = 5
noise_function = [sample_lpp_noise] * 6 #+ [sample_l2_noise]


Data = []
base = sample_on_sphere(n=300, d=3, seed=None)
noisy = base + np.random.normal(0, 0.1, base.shape)

All_result = []
for s in range(6):
    Proj_X = []
    for i in range(20):
        # Append the result of k_nearest_on_sphere to Data instead of overwriting it
        Data_i, _, _ = k_nearest_on_sphere(noisy, noisy[:, [i]], k=30)
        Data.append(Data_i)  # Append new data to the list
        mode = types[s]
        # Perform quadratic manifold factorization
        Q, Theta, c, taus, err, step = quadratic_manifold_factorization(
            Data_i, d=2, s=1,
            eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
            T=1000, tol=1e-7, mode=mode, delta=0.1, p=P_values[s]
        )
        # Optimize tau for the current sample
        proj, x = optimize_tau(x_i=noisy[:, [i]], c=c, Q=Q, Theta=Theta, d=2, s=1, eta_tau=1e-1, mode=mode, p=P_values[s])
        Proj_X.append(x)

    # Stack the projections into a result
    Result = np.column_stack(Proj_X)

    # Normalize and calculate the norm
    error = 0
    for i in range(20):
        error += (np.linalg.norm(Proj_X[i].reshape(-1,1)-base[:,i].reshape(-1,1))**2)/20
    #Result_norm = np.sum([np.linalg.norm(r - base)**2 for r in Proj_X])/Result.shape[1]
    All_result.append(error)
    print(f"Result norm: {error}, P_value:{P_values[s]}")
plt.figure()
plt.plot(P_values, All_result,lw=2)
plt.show()
print(All_result)

