import numpy as np
import matplotlib.pyplot as plt

'''
def sample_lpp_noise(n, d, p):
    """
    Sample n points in R^d from
    p(eps) ∝ exp(-||eps||_p^p)
    """
    eps = np.zeros((d, n))
    for k in range(d):
        eps[k,:] = sample_generalized_gaussian(n, p)
    return eps
'''

def sample_lpp_noise(n, d, p, random_state=None):
    """
    Sample n points in R^d from f(x) ∝ exp(-||x||_p^p).
    """
    rng = np.random.default_rng(random_state)

    # Sample magnitudes: |X|^p ~ Gamma(1/p, 1)
    G = rng.gamma(shape=1.0/p, scale=1.0, size=(n, d))
    mags = G ** (1.0 / p)

    # Random signs
    signs = rng.choice([-1.0, 1.0], size=(n, d))

    return (signs * mags).T


def sample_l2_noise(n, d, p=0):
    return sample_radial_laplace(n, d)


def sample_l1_noise(n, d, s = 1.0):
    return sample_laplace_factorized(n, d, scale=s)




def sample_radial_laplace(n, d):
    """
    Sample n points from radial Laplace distribution in R^d
    p(x) ∝ exp(-||x||_2)
    """

    # Step 1: sample random directions
    U = np.random.randn(n, d)
    U /= np.linalg.norm(U, axis=1, keepdims=True)

    # Step 2: sample radii
    R = np.random.gamma(shape=d, scale=1.0, size=n)

    # Step 3: combine
    return (U * R[:, None]).T





def generate_l2_noise(n, D, sigma=0.1):
    """
    Generate data for squared ℓ2 loss (Gaussian noise)

    f     : function mapping tau -> R^D
    taus  : (n, d) latent variables
    sigma : noise std
    """
    n = taus.shape[0]
    noise = sigma * np.random.randn(n, D)
    return noise


def sample_laplace_factorized(n, d, scale=1.0):
    """
    Sample n points in R^d from p(x) ∝ exp(-||x||_1)
    """
    return np.random.laplace(loc=0.0, scale=scale, size=(n, d)).T




def sample_generalized_gaussian(n, p):
    """
    Sample n i.i.d. from p(z) ∝ exp(-|z|^p)
    """
    # Gamma(shape=1/p, scale=1)
    u = np.random.gamma(shape=1.0/p, scale=1.0, size=n)
    
    # Random signs
    signs = np.random.choice([-1.0, 1.0], size=n)
    
    return signs * u ** (1.0 / p)


if __name__ == '__main__':
    n = 10000
    d = 2
    plt.figure()
    fig, axes = plt.subplots(1, 8, figsize=(12, 3))
    for i in range(6):
        X = sample_lpp_noise(n, d, p=i*0.16+1)
        axes[i].plot(np.sort(X.flatten()))
    X = sample_l2_noise(n, d)
    Y = sample_l1_noise(n, d)
    axes[6].plot(np.sort(X.flatten()))
    axes[7].plot(np.sort(Y.flatten()))
    plt.show()

