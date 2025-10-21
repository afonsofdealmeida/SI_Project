from utils.utils import *

def missing_values(df):
    """
    Preprocess the dataframe by imputing missing values, 
    separating variables into binary and continuous, 
    applying DecisionTree-based imputation, and creating logarithmic columns.
    """
    # Fill the column 'Dias_ate_morte' with the constant 800
    col_const_dias = 'Dias_ate_morte'
    df[col_const_dias] = df[col_const_dias].fillna(800)

    # Fill the column 'RecovTime' with the constant 400
    col_const_recov = 'RecovTime'
    df[col_const_recov] = df[col_const_recov].fillna(400)

    # Separate numerical columns
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Identify binary variables (only 0/1)
    binary_cols = [col for col in num_cols if set(df[col].dropna().unique()) <= {0, 1} 
                   and col not in [col_const_dias, col_const_recov]]

    # Continuous variables = remaining numerical (excluding constants and binaries)
    continuous_cols = [col for col in num_cols if col not in binary_cols + [col_const_dias, col_const_recov]]

    print(f"Continuous variables: {continuous_cols}")
    print(f"Binary variables: {binary_cols}")

    # Impute continuous variables with DecisionTreeRegressor
    if continuous_cols:
        imputer_cont = IterativeImputer(
            estimator=DecisionTreeRegressor(random_state=0),
            max_iter=30,
            random_state=0,
            min_value=0
        )
        df[continuous_cols] = imputer_cont.fit_transform(df[continuous_cols])

    # Impute binary variables with DecisionTreeClassifier
    if binary_cols:
        imputer_bin = IterativeImputer(
            estimator=DecisionTreeClassifier(random_state=0),
            max_iter=30,
            random_state=0
        )
        df[binary_cols] = imputer_bin.fit_transform(df[binary_cols])

    # Create logarithmic columns for 'Dias_ate_morte'
    for base, func in [('10', lambda x: np.log10(x+1))]:
        df[f'{col_const_dias}_log_{base}'] = func(df[col_const_dias])

    # Create logarithmic columns for 'RecovTime'
    for base, func in [('10', lambda x: np.log10(x+1))]:
        df[f'{col_const_recov}_log_{base}'] = func(df[col_const_recov])

    # Drop the original columns
    df = df.drop(columns=[col_const_dias, col_const_recov])

    return df
