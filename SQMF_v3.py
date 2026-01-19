import numpy as np


# ============================================================
# Helpers
# ============================================================

def sym(A: np.ndarray) -> np.ndarray:
    """Symmetric part."""
    return 0.5 * (A + A.T)


def retract_stiefel(Q: np.ndarray) -> np.ndarray:
    """
    QR-based retraction onto Stiefel manifold: columns become orthonormal.
    Ensures deterministic sign by forcing diag(R) nonnegative.
    """
    Qq, R = np.linalg.qr(Q)
    d = np.sign(np.diag(R))
    d[d == 0] = 1.0
    return Qq * d


def vech_upper(A: np.ndarray) -> np.ndarray:
    """Upper-triangular vectorization (including diagonal)."""
    idx = np.triu_indices(A.shape[0])
    return A[idx]


def build_M_tau(tau: np.ndarray) -> np.ndarray:
    """Matrix M_tau in R^{d(d+1)/2 x d}."""
    d = tau.size
    rows = []
    for i in range(d):
        for j in range(i, d):
            row = np.zeros(d)
            row[j] = tau[i]
            rows.append(row)
    return np.vstack(rows)


def build_N_tau(tau: np.ndarray) -> np.ndarray:
    """Matrix N_tau in R^{d(d+1)/2 x d}."""
    d = tau.size
    rows = []
    for i in range(d):
        for j in range(i, d):
            row = np.zeros(d)
            row[i] = tau[j]
            rows.append(row)
    return np.vstack(rows)


# ============================================================
# Loss factory
# ============================================================

def loss_factory(loss: str, kwargs: dict):
    """
    Returns a function f(r) -> (value, grad_r)
    """

    def l1(r):
        return np.linalg.norm(r, 1), np.sign(r)

    def l2(r):
        nrm = np.linalg.norm(r, 2)
        if nrm == 0:
            return 0.0, np.zeros_like(r)
        return nrm, r / nrm

    def l2_squared(r):
        return float(r @ r), 2.0 * r

    def lp_p(r):
        p = kwargs.get("p", 2)
        ar = np.abs(r)
        val = np.sum(ar ** p)
        grad = p * (ar ** (p - 1)) * np.sign(r)
        return float(val), grad

    def huber(r):
        delta = kwargs.get("delta", 1.0)
        nrm = np.linalg.norm(r, 2)
        if nrm <= delta:
            return 0.5 * nrm**2, r
        return delta * nrm - 0.5 * delta**2, delta * r / nrm

    def mahalanobis(r):
        M = kwargs["M"]
        val = float(np.sqrt(r.T @ M @ r))
        if val == 0:
            return 0.0, np.zeros_like(r)
        return val, (M @ r) / val

    losses = {
        "l1": l1,
        "l2": l2,
        "l2_squared": l2_squared,
        "lp_p": lp_p,
        "huber": huber,
        "mahalanobis": mahalanobis,
    }
    if loss not in losses:
        raise ValueError(f"Unknown loss '{loss}'. Options: {list(losses.keys())}")
    return losses[loss]


# ============================================================
# Core forward / objective utilities
# ============================================================

def forward_all(X, taus, c, U, V, Theta):
    """
    Compute predictions and residuals for all samples.

    Returns:
        F: (D, n) predictions
        R: (D, n) residuals F - X
        vech_list: list of vech(tau tau^T) for each i (length n)
        quad_list: list of Theta^T vech(tau tau^T) for each i (length n), each shape (s,)
    """
    D, n = X.shape
    d = U.shape[1]
    s = V.shape[1]

    vech_list = []
    quad_list = []
    F = np.empty((D, n), dtype=X.dtype)

    for i in range(n):
        tau = taus[i]
        vech_tt = vech_upper(np.outer(tau, tau))           # (d(d+1)/2,)
        quad = Theta.T @ vech_tt                           # (s,)
        f = c[:, 0] + U @ tau + V @ quad                   # (D,)
        F[:, i] = f
        vech_list.append(vech_tt)
        quad_list.append(quad)

    R = F - X
    return F, R, vech_list, quad_list


def objective_from_residuals(R, loss_func):
    """Sum loss over columns of R."""
    total = 0.0
    for i in range(R.shape[1]):
        total += loss_func(R[:, i])[0]
    return float(total)


def objective(X, taus, c, Q, Theta, d, s, loss_func):
    U = Q[:, :d]
    V = Q[:, d:d+s]
    _, R, _, _ = forward_all(X, taus, c, U, V, Theta)
    return objective_from_residuals(R, loss_func)


def objective_single(
    x_i: np.ndarray,
    tau: np.ndarray,
    c: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    Theta: np.ndarray,
    loss_func: callable
) -> float:
    r = eval_one_tau(tau, c, Theta, U, V) - x_i.reshape(-1,1)
    return float(loss_func(r)[0])


# ============================================================
# Main algorithm
# ============================================================
def eval_one_tau(tau: np.ndarray, c: np.ndarray, Theta: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    tau: (d,)
    c: (D,1)
    Theta: (m,s)
    U: (D,d), V: (D,s)
    returns f: (D,)
    """
    tau = np.asarray(tau).reshape(-1)
    vech_tt = vech_upper(np.outer(tau, tau))     # (m,)
    quad = Theta.T @ vech_tt                     # (s,)
    return (c[:, 0] + U @ tau + V @ quad).reshape(-1,1)          # (D,)

'''
def optimize_tau(
    x_i: np.ndarray,
    c: np.ndarray,
    Q: np.ndarray,
    Theta: np.ndarray,
    d: int,
    s: int,
    eta_tau: float,
    armijo_c: float = 1e-2,
    bt_max: int = 20,
    steps: int = 100,
    mode: str = "l1",
    **kwargs

) -> np.ndarray:
    """
    Gradient descent on tau with Armijo backtracking.

    Q: (D, d+s) Stiefel matrix
    """
    loss_func = loss_factory(mode, kwargs)
    tau = np.zeros(shape=(d,1))

    # Split Q
    U = Q[:, :d]        # (D,d)
    V = Q[:, d:d+s]     # (D,s)

    for _ in range(steps):
        # Jacobian J_tau = d f / d tau  (D,d)
        M_tau = build_M_tau(tau)
        N_tau = build_N_tau(tau)
        J_tau = U + V @ Theta.T @ (M_tau + N_tau)

        r = eval_one_tau(tau, c, Theta, U, V) - x_i
        _, g = loss_func(r)                     # g = d loss / d r (D,)
        grad_tau = J_tau.T @ g                  # (d,)
        gnorm = float(np.linalg.norm(grad_tau))
        if gnorm == 0.0:
            break

        desc_tau = -grad_tau
        obj0 = objective_single(x_i, tau, c, U, V, Theta, loss_func)

        eta = eta_tau
        tau_candidate = tau
        for _ in range(bt_max):
            tau_try = tau + eta * desc_tau
            obj_try = objective_single(x_i, tau_try, c, U, V, Theta, loss_func)

            # Armijo condition
            if obj_try <= obj0 - armijo_c * eta * (gnorm ** 2):
                tau_candidate = tau_try
                break
            eta *= 0.5

        tau = tau_candidate

    return tau
'''
def optimize_tau(
    x_i: np.ndarray,
    c: np.ndarray,
    Q: np.ndarray,
    Theta: np.ndarray,
    d: int,
    s: int,
    eta_tau: float,
    armijo_c: float = 1e-2,
    bt_max: int = 20,
    steps: int = 100,
    mode: str = "l1",
    tol: float = 1e-6,  # Tolerance for stopping condition
    **kwargs
) -> np.ndarray:
    """
    Gradient descent on tau with Armijo backtracking.

    Q: (D, d+s) Stiefel matrix
    """
    loss_func = loss_factory(mode, kwargs)
    tau = np.zeros(shape=(d, 1))

    # Split Q
    U = Q[:, :d]        # (D,d)
    V = Q[:, d:d+s]     # (D,s)

    last_obj = None

    for _ in range(steps):
        # Jacobian J_tau = d f / d tau  (D,d)
        M_tau = build_M_tau(tau)
        N_tau = build_N_tau(tau)
        J_tau = U + V @ Theta.T @ (M_tau + N_tau)

        r = eval_one_tau(tau, c, Theta, U, V) - x_i
        _, g = loss_func(r)                     # g = d loss / d r (D,)
        grad_tau = J_tau.T @ g                  # (d,)
        gnorm = float(np.linalg.norm(grad_tau))
        if gnorm == 0.0:
            break

        desc_tau = -grad_tau
        obj0 = objective_single(x_i, tau, c, U, V, Theta, loss_func)

        eta = eta_tau
        tau_candidate = tau
        for _ in range(bt_max):
            tau_try = tau + eta * desc_tau
            obj_try = objective_single(x_i, tau_try, c, U, V, Theta, loss_func)

            # Armijo condition
            if obj_try <= obj0 - armijo_c * eta * (gnorm ** 2):
                tau_candidate = tau_try
                break
            eta *= 0.5

        # Check if objective has changed significantly
        obj_change = abs(obj_try - obj0)
        if last_obj is not None and obj_change < tol:
            print(f"Stopping condition met. Objective change: {obj_change:.6f}")
            break
        last_obj = obj_try

        tau = tau_candidate

    return tau, eval_one_tau(tau, c, Theta, U, V)


def quadratic_manifold_factorization(
    X: np.ndarray,
    d: int,
    s: int,
    eta_Q: float,
    eta_Theta: float,
    eta_c: float,
    eta_tau: float,
    T: int = 100,
    tol: float = 1e-6,
    mode: str = "l1",
    armijo_c: float = 1e-2,
    bt_max: int = 20,
    set_Theta_zero: bool = False,
    **kwargs
):
    """
    Quadratic manifold factorization with robust loss and Stiefel constraint.

    Args:
        X: (D, n) data matrix
        d: linear latent dim
        s: quadratic latent dim
        eta_*: initial step sizes
        T: max iterations
        tol: stopping tolerance on objective change
        mode: loss name
        armijo_c: Armijo constant
        bt_max: max backtracking iterations

    Returns:
        Q, Theta, c, taus, err_list, step_sizes
    """
    D, n = X.shape
    loss_func = loss_factory(mode, kwargs)

    # ---- Initialization ----
    c = np.mean(X, axis=1, keepdims=True)  # (D,1)
    Xc = X - c
    U0, _, _ = np.linalg.svd(Xc, full_matrices=False)
    Q = U0[:, :d+s].copy()

    U = Q[:, :d]
    V = Q[:, d:d+s]

    # taus init (shape (d,))
    taus = [(U.T @ (X[:, i] - c[:, 0])).astype(X.dtype) for i in range(n)]

    Theta = np.zeros((d * (d + 1) // 2, s), dtype=X.dtype)

    err = []
    last_obj = None

    for t in range(T):
        # ---------- Forward + residuals ----------
        F, R, vech_list, quad_list = forward_all(X, taus, c, U, V, Theta)

        # loss + gradients wrt residual
        g_list = []
        total_loss = 0.0
        for i in range(n):
            val, g = loss_func(R[:, i])
            total_loss += val
            g_list.append(g)

        err.append(float(total_loss))

        # Stopping condition based on change in objective
        if last_obj is not None and abs(last_obj - total_loss) < tol:
            print(f"Stopping condition met at iteration {t}. Objective change: {abs(last_obj - total_loss):.6f}")
            break
        last_obj = total_loss

        # ========================================================
        # Update Q (Stiefel) with backtracking
        # ========================================================
        grad_Q = np.zeros_like(Q)

        # Each column block is [tau; quad] in R^{d+s}
        for i in range(n):
            block = np.concatenate([taus[i], quad_list[i]], axis=0)  # (d+s,)
            grad_Q += np.outer(g_list[i], block)                     # (D, d+s)

        # Riemannian gradient
        grad_Q = grad_Q - Q @ sym(Q.T @ grad_Q)
        desc_Q = -grad_Q

        obj0 = total_loss
        desc_norm2 = float(np.linalg.norm(desc_Q) ** 2)

        eta = eta_Q
        Q_candidate = Q
        for _ in range(bt_max):
            Q_try = retract_stiefel(Q + eta * desc_Q)
            obj_try = objective(X, taus, c, Q_try, Theta, d, s, loss_func)
            if obj_try <= obj0 - armijo_c * eta * desc_norm2:
                Q_candidate = Q_try
                obj0 = obj_try
                break
            eta *= 0.5

        Q = Q_candidate
        U = Q[:, :d]
        V = Q[:, d:d+s]
        
        # ========================================================
        # Update Theta with backtracking
        # ========================================================
        if set_Theta_zero:
            Theta = np.zeros_like(Theta)
        else:
            grad_Theta = np.zeros_like(Theta)
            Vt_g = [V.T @ g_list[i] for i in range(n)]  # each (s,)

            for i in range(n):
                grad_Theta += np.outer(vech_list[i], Vt_g[i])  # (m,s)

            desc_Theta = -grad_Theta
            obj_base = objective(X, taus, c, Q, Theta, d, s, loss_func)
            desc_norm2 = float(np.linalg.norm(desc_Theta) ** 2)

            eta = eta_Theta
            Theta_candidate = Theta
            for _ in range(bt_max):
                Theta_try = Theta + eta * desc_Theta
                obj_try = objective(X, taus, c, Q, Theta_try, d, s, loss_func)
                if obj_try <= obj_base - armijo_c * eta * desc_norm2:
                    Theta_candidate = Theta_try
                    break
                eta *= 0.5
            Theta = Theta_candidate
        
        # ========================================================
        # Update c with backtracking
        # ========================================================
        grad_c = np.sum(np.stack(g_list, axis=1), axis=1, keepdims=True)  # (D,1)
        desc_c = -grad_c

        obj_base = objective(X, taus, c, Q, Theta, d, s, loss_func)
        desc_norm2 = float(np.linalg.norm(desc_c) ** 2)

        eta = eta_c
        c_candidate = c
        for _ in range(bt_max):
            c_try = c + eta * desc_c
            obj_try = objective(X, taus, c_try, Q, Theta, d, s, loss_func)
            if obj_try <= obj_base - armijo_c * eta * desc_norm2:
                c_candidate = c_try
                break
            eta *= 0.5
        c = c_candidate

        # ========================================================
        # Update taus (per-sample backtracking)
        # ========================================================
        for i in range(n):
            tau = taus[i]
            M_tau = build_M_tau(tau)
            N_tau = build_N_tau(tau)
            # J_tau shape (D,d)
            J_tau = U + V @ Theta.T @ (M_tau + N_tau)

            grad_tau = J_tau.T @ g_list[i]  # (d,)
            desc_tau = -grad_tau
            desc_dec = float(np.linalg.norm(grad_tau))  # simple scalar for Armijo

            if desc_dec == 0:
                continue

            obj_i0 = objective_single(X[:, i], tau, c, U, V, Theta, loss_func)

            eta = eta_tau
            tau_candidate = tau
            for _ in range(bt_max):
                tau_try = tau + eta * desc_tau
                obj_try = objective_single(X[:, i], tau_try, c, U, V, Theta, loss_func)
                if obj_try <= obj_i0 - armijo_c * eta * desc_dec:
                    tau_candidate = tau_try
                    break
                eta *= 0.5

            taus[i] = tau_candidate

    return Q, Theta, c, taus, err, [eta_Q, eta_Theta, eta_c, eta_tau]

'''
def quadratic_manifold_factorization(
    X: np.ndarray,
    d: int,
    s: int,
    eta_Q: float,
    eta_Theta: float,
    eta_c: float,
    eta_tau: float,
    T: int = 100,
    tol: float = 1e-6,
    mode: str = "l1",
    armijo_c: float = 1e-2,
    bt_max: int = 20,
    **kwargs
):
    """
    Quadratic manifold factorization with robust loss and Stiefel constraint.

    Args:
        X: (D, n) data matrix
        d: linear latent dim
        s: quadratic latent dim
        eta_*: initial step sizes
        T: max iterations
        tol: stopping tolerance on objective change
        mode: loss name
        armijo_c: Armijo constant
        bt_max: max backtracking iterations

    Returns:
        Q, Theta, c, taus, err_list, step_sizes
    """
    D, n = X.shape
    loss_func = loss_factory(mode, kwargs)

    # ---- Initialization ----
    c = np.mean(X, axis=1, keepdims=True)  # (D,1)
    Xc = X - c
    U0, _, _ = np.linalg.svd(Xc, full_matrices=False)
    Q = U0[:, :d+s].copy()

    U = Q[:, :d]
    V = Q[:, d:d+s]

    # taus init (shape (d,))
    taus = [(U.T @ (X[:, i] - c[:, 0])).astype(X.dtype) for i in range(n)]

    Theta = np.zeros((d * (d + 1) // 2, s), dtype=X.dtype)

    err = []
    last_obj = None

    for t in range(T):
        # ---------- Forward + residuals ----------
        F, R, vech_list, quad_list = forward_all(X, taus, c, U, V, Theta)

        # loss + gradients wrt residual
        g_list = []
        total_loss = 0.0
        for i in range(n):
            val, g = loss_func(R[:, i])
            total_loss += val
            g_list.append(g)

        err.append(float(total_loss))

        if last_obj is not None and abs(last_obj - total_loss) < tol:
            break
        last_obj = total_loss

        # ========================================================
        # Update Q (Stiefel) with backtracking
        # ========================================================
        grad_Q = np.zeros_like(Q)

        # Each column block is [tau; quad] in R^{d+s}
        for i in range(n):
            block = np.concatenate([taus[i], quad_list[i]], axis=0)  # (d+s,)
            grad_Q += np.outer(g_list[i], block)                     # (D, d+s)

        # Riemannian gradient
        grad_Q = grad_Q - Q @ sym(Q.T @ grad_Q)
        desc_Q = -grad_Q

        obj0 = total_loss
        desc_norm2 = float(np.linalg.norm(desc_Q) ** 2)

        eta = eta_Q
        Q_candidate = Q
        for _ in range(bt_max):
            Q_try = retract_stiefel(Q + eta * desc_Q)
            obj_try = objective(X, taus, c, Q_try, Theta, d, s, loss_func)
            if obj_try <= obj0 - armijo_c * eta * desc_norm2:
                Q_candidate = Q_try
                obj0 = obj_try
                break
            eta *= 0.5

        Q = Q_candidate
        U = Q[:, :d]
        V = Q[:, d:d+s]
        
        # ========================================================
        # Update Theta with backtracking
        # ========================================================
        grad_Theta = np.zeros_like(Theta)
        Vt_g = [V.T @ g_list[i] for i in range(n)]  # each (s,)

        for i in range(n):
            grad_Theta += np.outer(vech_list[i], Vt_g[i])  # (m,s)

        desc_Theta = -grad_Theta
        obj_base = objective(X, taus, c, Q, Theta, d, s, loss_func)
        desc_norm2 = float(np.linalg.norm(desc_Theta) ** 2)

        eta = eta_Theta
        Theta_candidate = Theta
        for _ in range(bt_max):
            Theta_try = Theta + eta * desc_Theta
            obj_try = objective(X, taus, c, Q, Theta_try, d, s, loss_func)
            if obj_try <= obj_base - armijo_c * eta * desc_norm2:
                Theta_candidate = Theta_try
                break
            eta *= 0.5
        Theta = Theta_candidate
        
        # ========================================================
        # Update c with backtracking
        # ========================================================
        grad_c = np.sum(np.stack(g_list, axis=1), axis=1, keepdims=True)  # (D,1)
        desc_c = -grad_c

        obj_base = objective(X, taus, c, Q, Theta, d, s, loss_func)
        desc_norm2 = float(np.linalg.norm(desc_c) ** 2)

        eta = eta_c
        c_candidate = c
        for _ in range(bt_max):
            c_try = c + eta * desc_c
            obj_try = objective(X, taus, c_try, Q, Theta, d, s, loss_func)
            if obj_try <= obj_base - armijo_c * eta * desc_norm2:
                c_candidate = c_try
                break
            eta *= 0.5
        c = c_candidate

        # ========================================================
        # Update taus (per-sample backtracking)
        # ========================================================
        for i in range(n):
            tau = taus[i]
            M_tau = build_M_tau(tau)
            N_tau = build_N_tau(tau)
            # J_tau shape (D,d)
            J_tau = U + V @ Theta.T @ (M_tau + N_tau)

            grad_tau = J_tau.T @ g_list[i]  # (d,)
            desc_tau = -grad_tau
            desc_dec = float(np.linalg.norm(grad_tau))  # simple scalar for Armijo

            if desc_dec == 0:
                continue

            obj_i0 = objective_single(X[:, i], tau, c, U, V, Theta, loss_func)

            eta = eta_tau
            tau_candidate = tau
            for _ in range(bt_max):
                tau_try = tau + eta * desc_tau
                obj_try = objective_single(X[:, i], tau_try, c, U, V, Theta, loss_func)
                if obj_try <= obj_i0 - armijo_c * eta * desc_dec:
                    tau_candidate = tau_try
                    break
                eta *= 0.5

            taus[i] = tau_candidate

    return Q, Theta, c, taus, err, [eta_Q, eta_Theta, eta_c, eta_tau]
'''
