import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# Carregar dataset para contar missing values
df_percentage = pd.read_excel("../data/processed/cleaned_dataset.xlsx", header=0)

# Percentagem de missing values por feature
missing_percent_by_feature = (df_percentage.isna().sum() / len(df_percentage)) * 100
missing_percent_df = missing_percent_by_feature.reset_index()
missing_percent_df.columns = ['Feature', 'Missing_Percent']
missing_percent_df = missing_percent_df.sort_values(by='Missing_Percent', ascending=False)

print("Percentage of missing values per feature:")
print(missing_percent_df.to_string(index=False))

# Percentagem total de missing values
total_missing_percent = (df_percentage.isna().sum().sum() / (df_percentage.shape[0] * df_percentage.shape[1])) * 100

print(f"Percentage of missing values: {total_missing_percent:.2f}%")

# Carregar dataset para preencher missing values
df = pd.read_excel("../data/processed/cleaned_dataset2.xlsx", header=0)

# Preencher a coluna 'Dias_ate_morte' com a constante 800
col_const_dias = 'Dias_ate_morte'
df[col_const_dias] = df[col_const_dias].fillna(800)

# Preencher a coluna RecovTime com a constante 400
col_const_recov = 'RecovTime'
df[col_const_recov] = df[col_const_recov].fillna(400)

# Separar tipos de variáveis
num_cols = df.select_dtypes(include=['number']).columns.tolist()

# Identificar variáveis binárias (apenas 0/1)
binary_cols = [col for col in num_cols if set(df[col].dropna().unique()) <= {0, 1} 
               and col not in [col_const_dias, col_const_recov]]

# Variáveis contínuas = restantes numéricas (excluindo a constante e as binárias)
continuous_cols = [col for col in num_cols if col not in binary_cols + [col_const_dias, col_const_recov]]

print(f"Continuous features: {continuous_cols}")
print(f"Binary features: {binary_cols}")

# Imputar variáveis contínuas com DecisionTreeRegressor
if continuous_cols:
    imputer_cont = IterativeImputer(
        estimator=DecisionTreeRegressor(random_state=0),
        max_iter=30,
        random_state=0,
        min_value=0
    )
    df[continuous_cols] = imputer_cont.fit_transform(df[continuous_cols])

# Imputar variáveis binárias com DecisionTreeClassifier
if binary_cols:
    imputer_bin = IterativeImputer(
        estimator=DecisionTreeClassifier(random_state=0),
        max_iter=30,
        random_state=0
    )
    df[binary_cols] = imputer_bin.fit_transform(df[binary_cols])

# Criar colunas logarítmicas para 'Dias_ate_morte'
for base, func in [('e', np.log1p), ('2', lambda x: np.log2(x+1)), 
                   ('10', lambda x: np.log10(x+1)), ('20', lambda x: np.log(x+1)/np.log(20))]:
    df[f'{col_const_dias}_log_{base}'] = func(df[col_const_dias])
    
# Criar colunas logarítmicas para 'RecovTime'
for base, func in [('e', np.log1p), ('2', lambda x: np.log2(x+1)), 
                   ('10', lambda x: np.log10(x+1)), ('20', lambda x: np.log(x+1)/np.log(20))]:
    df[f'{col_const_recov}_log_{base}'] = func(df[col_const_recov])

# Apagar as colunas originais
df = df.drop(columns=[col_const_dias, col_const_recov])

# Exportar dataset final
df.to_excel("cleaned_dataset_tree.xlsx", index=False)