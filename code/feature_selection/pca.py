from utils.utils import *

def apply_pca(df, n_components=3, n_top_features=1, plot=True):
    """
    Apply PCA to a dataset, return transformed data, loadings, and optionally plot cumulative variance.
    
    Parameters:
    - df: pandas DataFrame with features only
    - n_components: number of principal components to keep
    - n_top_features: number of top contributing features to display per PC
    - plot: whether to plot cumulative explained variance
    
    Returns:
    - df_pca: DataFrame with principal components
    - loadings: DataFrame of PCA loadings (features x PCs)
    - explained_variance_ratio: array of explained variance ratios
    """

    # Convert DataFrame to NumPy array
    X = df.values

    # Standardize data — critical before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Create a new DataFrame for convenience
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])

    # Create loadings DataFrame
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=df.columns
    )

    # Print PCA summary
    print(f"Original feature count: {X.shape[1]}")
    print(f"Reduced to {df_pca.shape[1]} principal components")
    print("Explained variance ratios:", np.round(pca.explained_variance_ratio_, 4))
    print("Cumulative variance:", np.round(np.cumsum(pca.explained_variance_ratio_), 4))

    # Display top contributing features per principal component
    for pc in loadings.columns:
        top_features = loadings[pc].abs().sort_values(ascending=False).head(n_top_features)
        print(f"\nTop {n_top_features} features for {pc}:")
        for feature, value in top_features.items():
            print(f"  {feature:<25} | loading = {loadings.loc[feature, pc]:.4f}")

    # Plot cumulative explained variance
    if plot:
        plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         np.cumsum(pca.explained_variance_ratio_),
         marker='o')
    plt.xticks(range(1, len(pca.explained_variance_ratio_)+1))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.show()

    return df_pca, loadings, pca.explained_variance_ratio_
