from utils.utils import *
from clustering.utils import compute_sigmas, hard_labels, plot_fuzzy_clusters, plot_gaussian_membership

def fuzzy_cmeans(df_features, y, test_size=0.2, n_clusters=2, m=2.0, random_state=42, feature_indices=[0,2]):
    """
    Perform train-test split, standardize features, apply Fuzzy C-means clustering,
    compute cluster spreads, perform hard clustering, and plot membership functions.
    """
    # --- Train-test split ---
    Xtr, Xte, ytr, yte = train_test_split(
        df_features.values,
        y.values,
        test_size=test_size,
        random_state=random_state
    )

    # --- Standardize features ---
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)
    print(f"Training feature shape: {Xtr.shape}")

    # --- Concatenate target with features for clustering ---
    Xexp = np.concatenate([Xtr, ytr.reshape(-1, 1)], axis=1)
    Xexp_T = Xexp.T  # skfuzzy expects shape (features, samples)

    # --- Fuzzy C-means clustering ---
    centers, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data=Xexp_T,
        c=n_clusters,
        m=m,
        error=0.005,
        maxiter=1000,
        init=None,
        seed=random_state
    )
    print("Fuzzy partition coefficient (FPC):", fpc)

    # --- Compute sigmas and hard labels ---
    sigmas = compute_sigmas(Xexp, centers, u, m)
    cluster_labels = hard_labels(u)

    # --- Plots ---
    plot_fuzzy_clusters(Xexp, u, cluster_labels, feature_indices)
    plot_gaussian_membership(Xexp, centers, sigmas, feature_index=feature_indices[1])

    return Xtr, Xte, ytr, yte, centers, u, fpc, sigmas, cluster_labels

