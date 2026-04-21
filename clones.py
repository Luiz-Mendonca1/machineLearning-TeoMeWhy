# %%

import pandas as pd

df = pd.read_parquet("data/dados_clones.parquet")

df['General Jedi encarregado'].unique()


# %%

features = ['Massa(em kilos)', 'Estatura(cm)']

df.groupby('Status ')[features].mean()
#encontra diferenca entre apto ou defeituoso de Apto	83.765887	180.605545; Defeituoso	83.200134	180.400186