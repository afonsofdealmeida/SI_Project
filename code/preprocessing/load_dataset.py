from utils.utils import *

def load_and_clean_dataset(filepath, sheet_name='Baseline_TP1'):
    """
    Load the Excel dataset, set the first row as column names, 
    drop unnecessary columns, and remove inconsistent rows.
    
    Parameters:
    - filepath: path to the Excel file
    - sheet_name: sheet name to read from (default: 'Baseline_TP1')
    
    Returns:
    - df: cleaned pandas DataFrame
    """

    # Load the dataset and make the first row the column names
    df = pd.read_excel(filepath, header=0, sheet_name=sheet_name)

    # Columns to remove
    cols_to_remove = [
        "PROCESSO",
        "N",
        "DATA_AVALIACAO",
        "DATA_NASCIMENTO",
        "DATA _ADMISSAO_HOSPITAL",
        "DATA_ADMISSAO_UCI",
        "DATA_ALTA_UCI",
        "DATA_ALTA_HOSPITAL",
        "Data_Obito"
    ]

    # Drop the unnecessary columns
    df = df.drop(columns=cols_to_remove)

    # Drop the fourth patient (index 3) due to inconsistent values
    df = df.drop(index=3)

    return df