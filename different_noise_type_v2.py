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
        eta_Q=1e-2, eta_Theta=1e-2, eta_c=1e-2, eta_tau=1e-2,
        T=1000, tol=1e-10, mode=mode, delta=0.1, p=p_value
    )

    taus = np.asarray(taus).reshape(-1, 1)
    U = Q[:, [0]]
    V = Q[:, [1]]

    result = eval_model(c, U, V, Theta, taus)

    t_mid = np.linspace(taus.min(), taus.max(), 50).reshape(-1, 1)
    result2 = eval_model(c, U, V, Theta, t_mid)

    return result, result2, err, step

# ---------------- Experiment setup ----------------

types = ['lp_p'] * 5 + ['l2']
labels = [
    r'$\ell_p^p, p=1.0$', r'$\ell_p^p, p=1.25$', r'$\ell_p^p, p=1.5$',
    r'$\ell_p^p, p=1.75$', r'$\ell_p^p, p=2.0$', r'$\ell_2$'
]
P_values = [1, 1.25, 1.5, 1.75, 2, 2]

partition = 4
noise_function = [sample_lpp_noise] * 5 + [sample_l2_noise]

Data = []
for f in range(6):
    k = 0
    base = generate_data(
        start=2*np.pi/partition*k,
        end=2*np.pi/partition*(k+1),
        num=30
    )
    noisy = base + noise_function[f](n=30, d=2, p=P_values[f]) * 0.1
    Data.append(noisy)

# ---------------- Visualization ----------------

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
errors = []
steps = []

for i in range(6):
    XY = Data[i]
    result, result2, err, step = fit_and_reconstruct(XY, types[i], P_values[i])
    ax = axes[i // 3, i % 3]
    ax.tick_params(labelsize=16)

    t = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), '--', lw=1.6)

    ax.scatter(XY[0], XY[1], s=16, label='Noisy Data')
    ax.scatter(result[0], result[1], s=16, marker='x', label='Projection')
    ax.plot(result2[0], result2[1], color='red', lw=2, label='Fitted Curve')

    for k in range(XY.shape[1]):
        ax.annotate(
            '', xytext=(XY[0, k], XY[1, k]),
            xy=(result[0, k], result[1, k]),
            arrowprops=dict(arrowstyle='->', lw=1, color='black')
        )

    ax.set_title(labels[i], fontsize=16)
    errors.append(err)
    steps.append(step)

plt.tight_layout()
plt.savefig("robustness2.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ---------------- Learning rate ----------------

plt.figure()
plt.tick_params(labelsize=16)
for i in range(5):
    plt.scatter([1, 2, 3, 4], steps[i], marker='x')
plt.legend(labels)
plt.xlabel('Q, Theta, c, tau')
plt.title('Final Step Size')
plt.savefig("learning_rate2.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ---------------- Error curves ----------------

plt.figure()
plt.tick_params(labelsize=16)
for i in range(5):
    e = errors[i]
    plt.plot(np.arange(1, len(e) + 1), e)
plt.legend(labels)
plt.xlabel('Iteration')
plt.title('Approximation Error With Iterations')
plt.savefig("error2.pdf", dpi=300, bbox_inches="tight")
plt.show()