from utils.utils import *

def encode(df, embedding_dim=4, seed=42):
    """
    Encode 'SEXO' column, handle missing 'Agente' entries, 
    create embeddings for 'Agente', and add them as new columns.
    
    Parameters:
    - df: pandas DataFrame
    - embedding_dim: dimension of the agent embeddings
    - seed: random seed for reproducibility
    
    Returns:
    - df: DataFrame with agent embeddings added and 'Agente' dropped
    """

    # Encode 'SEXO' column: 'F' -> 0, 'M' -> 1
    df['SEXO'] = df['SEXO'].map({'F': 0, 'M': 1})

    # Replace missing or zero entries in 'Agente' and drop rows with NaN
    df['Agente'] = df['Agente'].replace('0', np.nan)
    df = df.dropna(subset=['Agente'])

    # Create a lookup for embeddings
    unique_agents = df['Agente'].unique()
    rng = np.random.default_rng(seed=seed)
    agent_embeddings = {agent: rng.normal(0, 1, embedding_dim) for agent in unique_agents}

    # Add embedding columns to dataframe
    for i in range(embedding_dim):
        df[f'emb_{i}'] = df['Agente'].map(lambda x: agent_embeddings[x][i])

    # Drop the original 'Agente' column
    df = df.drop(columns=['Agente'])

    return df
