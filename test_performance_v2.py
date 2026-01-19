import numpy as np
from SQMF_v3 import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_generator import *
import os
import pickle

mpl.rcParams['text.usetex'] = True
os.makedirs("modified/figures", exist_ok=True)

# ---------------- Data generation ----------------

def generate_data(start, end, num):
    """Generate 2D data (cosine and sine)."""
    t = np.linspace(start, end, num)
    x = np.cos(t)
    y = np.sin(t)
    return np.vstack((x, y))

def eval_model(c, U, V, Theta, taus):
    """Evaluate the model by computing the output for given tau values."""
    taus = np.asarray(taus)
    out = [c.ravel() + U @ tau + V @ (Theta.T @ vech_upper(np.outer(tau, tau))) for tau in taus]
    return np.column_stack(out)

def fit_and_reconstruct(XY, mode, p_value):
    """Perform quadratic manifold factorization and return results."""
    Q, Theta, c, taus, err, step = quadratic_manifold_factorization(
        XY, d=1, s=1,
        eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
        T=1000, tol=1e-10, mode=mode, delta=0.1, p=p_value
    )

    taus = np.asarray(taus).reshape(-1, 1)
    U, V = Q[:, [0]], Q[:, [1]]
    result = eval_model(c, U, V, Theta, taus)

    t_mid = np.linspace(taus.min(), taus.max(), 50).reshape(-1, 1)
    result2 = eval_model(c, U, V, Theta, t_mid)

    return result, result2, err, step

def sample_on_sphere(n, d, seed=None):
    """Uniformly sample n points on the sphere S^{d-1}."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.T  # shape (d, n)

def k_nearest_on_sphere(X, x0, k):
    """Return the k nearest neighbors of x0 from X (columns)."""
    x0 = x0 / np.linalg.norm(x0)
    dists = np.linalg.norm(X - x0, axis=0)
    idx = np.argsort(dists)[:k]
    return X[:, idx], idx, dists[idx]

def refine_via_qmf(noisy, N=20, mode='lp_p', p_value=1, neig=30):
    """Refine projections via quadratic manifold factorization."""
    Proj_X = []
    for i in range(N):
        Data_i, _, _ = k_nearest_on_sphere(noisy, noisy[:, [i]], k=neig)
        # Perform quadratic manifold factorization
        Q, Theta, c, taus, err, step = quadratic_manifold_factorization(
            Data_i, d=2, s=1,
            eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
            T=1000, tol=1e-7, mode=mode, delta=0.1, p=p_value
        )
        proj, x = optimize_tau(x_i=noisy[:, [i]], c=c, Q=Q, Theta=Theta, d=2, s=1, eta_tau=1e-1, mode=mode, p=p_value)
        Proj_X.append(x)

    return np.column_stack(Proj_X)

def compute_error_per_sample(Proj_X, base, N=20):
    # Calculate error for the projections
    error, error1 = [], []
    for i in range(N):
        temp = Proj_X[:, i].reshape(-1, 1)
        error.append(np.linalg.norm(temp - base[:, i].reshape(-1, 1))**2)
        error1.append(np.linalg.norm(temp - temp / np.linalg.norm(temp))**2)
    return error, error1

def run_experiment():
    """Run the full experiment and store the results."""
    types = ['lp_p'] * 5 + ['l2']
    P_values = [1, 1.25, 1.5, 1.75, 2, 2]
    
    Data = []
    base = sample_on_sphere(n=300, d=3, seed=None)
    All_result_truth, All_result_false = [], []

    for t in range(5):  # Vary noise levels
        result_noise_level_truth, result_noise_level_false = [], []
        noisy = base + np.random.normal(0, 0.05 * t, base.shape)

        for s in range(6):  # Iterate over modes (lp_p and l2)
            Proj_X = refine_via_qmf(noisy, N=20, mode=types[s], p_value=P_values[s], neig=30)
            error, error1 = compute_error_per_sample(Proj_X, base, N=20)
            result_noise_level_truth.append(error)
            result_noise_level_false.append(error1)
            print(f"Result norm: {error}, P_value:{P_values[s]}")

        All_result_truth.append(result_noise_level_truth)
        All_result_false.append(result_noise_level_false)

    with open('data_v2.pkl', 'wb') as file:
        pickle.dump([All_result_false, All_result_truth], file)


# ---------------- Experiment setup ----------------
if __name__ == '__main__':
    if True:
        run_experiment()
    else:
        xl = [r'$p=1.00$', r'$1.25$', r'$1.50$', r'$1.75$', r'$2.00$', r'$\ell_2$']
        with open('data_v2.pkl', 'rb') as file:
            data = pickle.load(file)

            # Create the plot
            fig, axe = plt.subplots(2, 2, figsize=(15, 8))
            data_print = np.zeros((4, 6))

            for i in range(4):     
                To_ana, La = [], []
                ax = axe[i // 2, i % 2]
                for k in range(6):
                    to_analysis = np.array(data[1][i + 1][k])
                    data_print[i, k] = to_analysis.mean()
                    To_ana.append(to_analysis)
                    La.append(k + 1)

                ax.boxplot(To_ana, positions=La, widths=0.6, showfliers=False)
                ax.set_xticks(np.arange(1, 7))
                ax.set_xticklabels(xl, fontsize=16)
                ax.set_title(r'$\sigma=' + f'{0.05 * (i + 1):.2g}$', fontsize=16)
                ax.set_xlabel('Different $\ell_p^p$ Values', fontsize=16)
                ax.set_ylabel('Error', fontsize=16)

            plt.tight_layout()
            plt.savefig('sphere_mean_var.pdf', dpi=300)
            plt.show()
            np.set_printoptions(precision=3, suppress=True)
            print(data_print)
