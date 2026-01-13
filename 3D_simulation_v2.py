import numpy as np
from back_tracking_SQMF_2 import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_generator import *
import os

mpl.rcParams['text.usetex'] = True
os.makedirs("modified/figures", exist_ok=True)

# ---------------- Utilities ----------------

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

def eval_model(c, U, V, Theta, taus):
    """Vectorized evaluation of quadratic manifold."""
    taus = np.asarray(taus)
    out = []
    for tau in taus:
        vech_tt = vech_upper(np.outer(tau, tau))
        f = c.ravel() + U @ tau + V @ (Theta.T @ vech_tt)
        out.append(f)
    return np.column_stack(out)

def fit_and_reconstruct(XY, mode, p_value):
    Q, Theta, c, taus, err, step = quadratic_manifold_factorization(
        XY, d=2, s=1,
        eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
        T=1000, tol=1e-7, mode=mode, delta=0.1, p=p_value
    )

    taus = np.asarray(taus)
    result = eval_model(c, Q[:, :2], Q[:, [2]], Theta, taus)

    # build grid in latent space
    x = np.linspace(taus[:, 0].min(), taus[:, 0].max(), 20)
    y = np.linspace(taus[:, 1].min(), taus[:, 1].max(), 20)
    Xg, Yg = np.meshgrid(x, y)
    XY_grid = np.vstack([Xg.ravel(), Yg.ravel()])

    taus_grid = [XY_grid[:, i] for i in range(XY_grid.shape[1])]
    result2 = eval_model(c, Q[:, :2], Q[:, [2]], Theta, taus_grid)

    return Xg, Yg, result, result2, err

# ---------------- Experiment Setup ----------------

types = ['lp_p'] * 5 + ['l2']
labels = [
    r'$\ell_p^p, p=1.0$', r'$\ell_p^p, p=1.25$', r'$\ell_p^p, p=1.5$',
    r'$\ell_p^p, p=1.75$', r'$\ell_p^p, p=2.0$', r'$\ell_2$'
]
P_values = [1, 1.25, 1.5, 1.75, 2, 2]

n, d, k = 300, 3, 30
Data = []

for i in range(6):
    data_on_sphere = sample_on_sphere(n, d)
    Xk, _, _ = k_nearest_on_sphere(
        data_on_sphere, np.array([1, 0, 1]).reshape(-1, 1), k
    )
    Xk += sample_lpp_noise(n=k, d=3, p=P_values[i]) * 0.1
    Data.append(Xk)

# ---------------- Visualization ----------------

fig, axes = plt.subplots(2, 3, figsize=(12, 8), subplot_kw={'projection': '3d'})
errors = []

# unit sphere for reference
phi, theta = np.meshgrid(
    np.linspace(0, np.pi, 20),
    np.linspace(0, 2*np.pi, 20)
)
Xs = np.sin(phi) * np.cos(theta)
Ys = np.sin(phi) * np.sin(theta)
Zs = np.cos(phi)

for i in range(6):
    XY = Data[i]
    Xg, Yg, result, result2, err = fit_and_reconstruct(XY, types[i], P_values[i])
    ax = axes[i // 3, i % 3]
    ax.tick_params(labelsize=12)

    ax.scatter(XY[0], XY[1], XY[2], s=16, label='Noisy Data')

    ns = int(np.sqrt(result2.shape[1]))
    ax.plot_surface(
        result2[0].reshape(ns, ns),
        result2[1].reshape(ns, ns),
        result2[2].reshape(ns, ns),
        color='blue', alpha=0.4
    )

    ax.plot_surface(Xs, Ys, Zs, color='cyan', alpha=0.2)
    ax.set_title(labels[i], fontsize=16)

    errors.append(err)

plt.tight_layout()
plt.savefig("robustness2_3d.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ---------------- Error Curves ----------------

plt.figure()
for i, e in enumerate(errors):
    plt.plot(np.arange(1, len(e) + 1), e, label=labels[i])

plt.legend()
plt.xlabel('Iteration')
plt.title('Approximation Error With Iterations')
plt.savefig("error2_3d.pdf", dpi=300, bbox_inches="tight")
plt.show()