from utils.utils import *

def correlation_filter(df, threshold):
    """
    Filter out features that are highly correlated above a given threshold.
    Keeps the feature with the highest total correlation and drops others.
    
    Parameters:
    - df: pandas DataFrame with numerical features
    - threshold: correlation threshold for filtering
    
    Returns:
    - reduced_df: DataFrame with selected features
    - to_drop: list of features that were dropped
    """

    # Compute absolute correlation matrix
    corr_matrix = df.corr().abs()
    
    # Compute total correlation sum for each feature
    corr_sum = corr_matrix.sum(axis=0)
    
    # Ignore self-correlations (diagonal)
    np.fill_diagonal(corr_matrix.values, 0)
    
    # Track features to drop
    to_drop = set()
    
    # Iterate over feature pairs
    for col in corr_matrix.columns:
        if col in to_drop:
            continue
        
        # Find all features correlated with 'col' above threshold
        high_corr = corr_matrix.index[corr_matrix[col] > threshold].tolist()
        
        for other in high_corr:
            if other not in to_drop and other != col:
                # Compare total correlation sums
                if corr_sum[col] >= corr_sum[other]:
                    to_drop.add(col)
                    break
                else:
                    to_drop.add(other)
    
    # Return reduced dataframe
    reduced_df = df.drop(columns=list(to_drop))
    return reduced_df, list(to_drop)
