import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from SQMF_v3 import *
import matplotlib.pyplot as plt

def normalize_img(img):
    img = img - img.min()
    img = img / (img.max() + 1e-8)
    return img

# -------------------
# Load MNIST 4/9
# -------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
images, labels = next(iter(train_loader))

mask = (labels == 4) | (labels == 9)
images_47 = images[mask][:40]   # 40 samples
labels_47 = labels[mask][:40]

# flatten to vectors
X = images_47.view(images_47.shape[0], -1).numpy().T   # (784, n)

d = 2      # linear latent dim
s = 20     # quadratic latent dim

# -------------------
# 1️⃣ L2 mode
# -------------------
Q_lin_l2, Theta_lin_l2, c_lin_l2, taus_lin_l2, _, _ = quadratic_manifold_factorization(
    X, d=d, s=s,
    eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
    T=1000, tol=1e-10, mode='l2', delta=0.1, p=2,
    set_Theta_zero=True
)
U_lin_l2, V_lin_l2 = Q_lin_l2[:, :d], Q_lin_l2[:, d:d+s]
F_lin_l2, _, _, _ = forward_all(X, taus_lin_l2, c_lin_l2, U_lin_l2, V_lin_l2, Theta_lin_l2)

Q_quad_l2, Theta_quad_l2, c_quad_l2, taus_quad_l2, _, _ = quadratic_manifold_factorization(
    X, d=d, s=s,
    eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
    T=1000, tol=1e-10, mode='l2', delta=0.1, p=2,
    set_Theta_zero=False
)
U_quad_l2, V_quad_l2 = Q_quad_l2[:, :d], Q_quad_l2[:, d:d+s]
F_quad_l2, _, _, _ = forward_all(X, taus_quad_l2, c_quad_l2, U_quad_l2, V_quad_l2, Theta_quad_l2)

# -------------------
# 2️⃣ LP_2 mode
# -------------------
Q_lin_lp, Theta_lin_lp, c_lin_lp, taus_lin_lp, _, _ = quadratic_manifold_factorization(
    X, d=d, s=s,
    eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
    T=1000, tol=1e-10, mode='lp_p', delta=0.1, p=2,
    set_Theta_zero=True
)
U_lin_lp, V_lin_lp = Q_lin_lp[:, :d], Q_lin_lp[:, d:d+s]
F_lin_lp, _, _, _ = forward_all(X, taus_lin_lp, c_lin_lp, U_lin_lp, V_lin_lp, Theta_lin_lp)

Q_quad_lp, Theta_quad_lp, c_quad_lp, taus_quad_lp, _, _ = quadratic_manifold_factorization(
    X, d=d, s=s,
    eta_Q=1e-1, eta_Theta=1e-1, eta_c=1e-1, eta_tau=1e-1,
    T=1000, tol=1e-10, mode='lp_p', delta=0.1, p=2,
    set_Theta_zero=False
)
U_quad_lp, V_quad_lp = Q_quad_lp[:, :d], Q_quad_lp[:, d:d+s]
F_quad_lp, _, _, _ = forward_all(X, taus_quad_lp, c_quad_lp, U_quad_lp, V_quad_lp, Theta_quad_lp)

# -------------------
# Visualization: each method in a row
# -------------------
methods = ["Original", "Linear (L2)", "Quadratic (L2)", "Linear (LP₂)", "Quadratic (LP₂)"]
images_list = [
    X.T.reshape(-1, 28, 28),
    F_lin_l2.T.reshape(-1, 28, 28),
    F_quad_l2.T.reshape(-1, 28, 28),
    F_lin_lp.T.reshape(-1, 28, 28),
    F_quad_lp.T.reshape(-1, 28, 28)
]

n_methods = len(methods)
n_show = 16

fig, axes = plt.subplots(
    n_methods,
    n_show,
    figsize=(2*n_show, 2*n_methods),
    constrained_layout=True   # 比 tight_layout 更不留白
)

for row in range(n_methods):
    for col in range(n_show):
        axes[row, col].imshow(
            normalize_img(images_list[row][col]),
            cmap="gray"
        )
        axes[row, col].axis("off")

    axes[row, 0].set_ylabel(
        methods[row],
        rotation=0,
        labelpad=40,
        fontsize=12,
        va="center"
    )

# ---- save with high dpi & no blank margins ----
plt.savefig(
    "sqmf_comparison.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)

plt.show()