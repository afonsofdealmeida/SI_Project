from utils.utils import *

def compute_adaptive_correlation_threshold(df, k=2):
    """
    Compute an adaptive correlation threshold based on the mean and standard deviation
    of absolute pairwise correlations in the dataframe.
    
    Parameters:
    - df: pandas DataFrame with numerical columns
    - k: multiplier for the standard deviation (default: 3)
    
    Returns:
    - threshold: adaptive correlation threshold
    - mean_corr: mean of upper-triangle absolute correlations
    - std_corr: standard deviation of upper-triangle absolute correlations
    """

    # Compute absolute correlation matrix
    corr_matrix = df.corr().abs()

    # Take the upper triangle (exclude duplicates and diagonal)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_values = corr_matrix.where(mask).stack().values

    # Compute mean and standard deviation
    mean_corr = np.mean(corr_values)
    std_corr = np.std(corr_values)

    # Choose threshold adaptively
    threshold = mean_corr + k * std_corr

    print(f"Mean correlation: {mean_corr:.3f}")
    print(f"Std deviation: {std_corr:.3f}")
    print(f"Adaptive threshold (k={k}): {threshold:.3f}")

    return threshold, mean_corr, std_corr
