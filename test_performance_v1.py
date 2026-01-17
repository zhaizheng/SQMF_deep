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

def run_experiment():
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

    All_result_truth = []
    All_result_false = []
    for t in range(5):
        result_noise_level_truth = []
        result_noise_level_false = []
        noisy = base + np.random.normal(0, 0.05*t, base.shape)
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
            error = []
            error1 = []
            for i in range(20):
                temp = Proj_X[i].reshape(-1,1)
                error.append(np.linalg.norm(temp - base[:,i].reshape(-1,1))**2) 
                error1.append(np.linalg.norm(temp-temp/np.linalg.norm(temp))**2)
            #Result_norm = np.sum([np.linalg.norm(r - base)**2 for r in Proj_X])/Result.shape[1]
            result_noise_level_truth.append(error)
            result_noise_level_false.append(error1)
            print(f"Result norm: {error}, P_value:{P_values[s]}")
        All_result_truth.append(result_noise_level_truth)
        All_result_false.append(result_noise_level_false)

    with open('data.pkl','wb') as file:
        pickle.dump([All_result_false, All_result_truth],file)

# ---------------- Experiment setup ----------------
if __name__ == '__main__':
    
    if False:
        run_experiment()
    else:
        xl = [r'$p=1.00$', r'$1.25$', r'$1.50$', r'$1.75$', r'$2.00$', r'$\ell_2$']
        with open('data.pkl', 'rb') as file:
            data = pickle.load(file)

            # Create the plot
            fig, axe = plt.subplots(2, 2, figsize=(15, 8))

            # Loop through the data
            for i in range(4):     
                # Collect data to be plotted in a boxplot format
                To_ana = []
                La = []
                ax = axe[i // 2, i % 2]
                for k in range(6):
                    to_analysis = np.array(data[1][i + 1][k])
                    To_ana.append(to_analysis)
                    La.append(k + 1)
                    # Create a boxplot for each set
                ax.boxplot(To_ana, positions=La, widths=0.6, showfliers=False)
                
                # Set x-ticks and labels for the boxplot
                ax.set_xticks(np.arange(1, 7))
                ax.set_xticklabels(xl, fontsize=16)  # Set the font size for x-tick labels

                # Title and labels
                ax.set_title(r'$\sigma=' + f'{0.05*(i+1):.2g}$', fontsize=16)  # Set the font size for title
                ax.set_xlabel('Different $\ell_p^p$ Values', fontsize=16)  # Set font size for x-axis label
                ax.set_ylabel('Error', fontsize=16)  # Set font size for y-axis label

            # Show the plot
            plt.tight_layout()
            plt.savefig('sphere_mean_var.pdf', dpi=300)
            plt.show()

            '''
            xl = [r'$\ell_p^p, p=1.0$', r'$\ell_p^p, p=1.25$', r'$\ell_p^p, p=1.5$',
            r'$\ell_p^p, p=1.75$', r'$\ell_p^p, p=2.0$', r'$\ell_2$']
        
            with open('data.pkl', 'rb') as file:
                data = pickle.load(file)
            
            # Create two subplots: one for data[0] and one for data[1]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns
            
            # Plot data[0] on the left subplot (ax1)
            for i in range(5):
                Mean = []
                Vari = []
                for k in range(6):
                    to_analysis = np.array(data[0][i][k])
                    Mean.append(to_analysis.mean())
                    Vari.append(to_analysis.var())
                ax1.plot(Mean)

            ax1.set_title('Fake')
            ax1.set_xticks(range(6))  # Adjust according to the data
            ax1.set_xticklabels(xl)   # Set x-axis labels for data[0]
            ax1.legend([f"{0.05*(i+1):.2f}" for i in range(5)])  # Set legend for data[0]
                        
            # Adjust the layout for better spacing
            plt.tight_layout()
            
            # Show the plot
            plt.show()
            '''
            

#with open('data.pkl','rb') as file:
#    data = pickle.load(file)

#print(All_result)

