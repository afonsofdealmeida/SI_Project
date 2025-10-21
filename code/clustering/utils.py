from utils.utils import *

def compute_sigmas(Xexp, centers, u, m=2.0):
    """
    Compute cluster spreads (sigmas) for Fuzzy C-means clusters.
    """
    n_clusters = centers.shape[0]
    sigmas = []
    for j in range(n_clusters):
        u_j = u[j, :] ** m
        var_j = np.average((Xexp - centers[j])**2, axis=0, weights=u_j)
        sigma_j = np.sqrt(var_j)
        sigmas.append(sigma_j)
    return np.array(sigmas)

def hard_labels(u):
    """
    Compute hard cluster labels from fuzzy membership matrix.
    """
    return np.argmax(u, axis=0)

def plot_fuzzy_clusters(Xexp, u, cluster_labels, feature_indices=[0,2]):
    """
    Scatter plot of fuzzy clustering with transparency proportional to membership.
    """
    n_clusters = u.shape[0]
    plt.figure(figsize=(8,6))
    for j in range(n_clusters):
        plt.scatter(
            Xexp[cluster_labels == j, feature_indices[0]],
            Xexp[cluster_labels == j, feature_indices[1]],
            alpha=u[j, :],
            label=f'Cluster {j}'
        )
    plt.title("Fuzzy C-Means Clustering (with membership degree)")
    plt.xlabel(f"Feature {feature_indices[0]+1}")
    plt.ylabel(f"Feature {feature_indices[1]+1}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_gaussian_membership(Xexp, centers, sigmas, feature_index=0):
    """
    Plot Gaussian membership curves for a selected feature.
    """
    n_clusters = centers.shape[0]

    def gaussian(x, mu, sigma):
        return np.exp(-0.5 * ((x - mu)/sigma)**2)

    lin = np.linspace(np.min(Xexp[:, feature_index])-1, np.max(Xexp[:, feature_index])+1, 500)
    plt.figure(figsize=(8,6))
    for j in range(n_clusters):
        y_curve = gaussian(lin, centers[j, feature_index], sigmas[j, feature_index])
        plt.plot(lin, y_curve, label=f"Gaussian μ={np.round(centers[j, feature_index],2)}, σ={np.round(sigmas[j, feature_index],2)}")
    plt.title(f"Projection of the membership functions on Feature {feature_index+1}")
    plt.xlabel(f"Feature {feature_index+1}")
    plt.ylabel("Degree of Membership")
    plt.legend()
    plt.grid(True)
    plt.show()