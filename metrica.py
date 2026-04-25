# %%
import pandas as pd

url = "https://docs.google.com/spreadsheets/d/1YQBQ3bu1TCmgrRch1gzW5O4Jgc8huzUSr7VUkxg0KIw/export?gid=283387421&format=csv"

df = pd.read_csv(url)
df.head()
# %%

df = df.replace({'Sim': 1, 'Não': 0})
df.head()

num_vars = [
    'Curte games?',
    'Curte futebol?',
    'Curte livros?',
    'Curte jogos de tabuleiro?',
    'Curte jogos de fórmula 1?',
    'Curte jogos de MMA?',
    'Idade',
]

dummy_vars = [
    "Como conheceu o Téo Me Why?",
    "Quantos cursos acompanhou do Téo Me Why?",
    "Estado que mora atualmente",
    "Área de Formação",
    "Tempo que atua na área de dados",
    "Posição da cadeira (senioridade)",
]

df_analise = pd.get_dummies(df[dummy_vars]).astype(int)
df_analise[num_vars] = df[num_vars].copy()
df_analise['pessoa feliz'] = df['Você se considera uma pessoa feliz?'].copy()
df_analise

#%%

features = df_analise.columns[:-1].tolist()
x=df_analise[features]
y=df_analise['pessoa feliz']

from sklearn import tree

arvore = tree.DecisionTreeClassifier(random_state=42, 
                                     min_samples_leaf=5
                                     )

arvore.fit(x, y)

#%%

arvore_predict = arvore.predict(x)
arvore_predict

df_predict = df_analise[['pessoa feliz']]
df_predict['predict_arvore'] = arvore_predict
df_predict

#%%

#acuracia
(df_predict['pessoa feliz'] == df_predict['predict_arvore']).mean()
# %%

pd.crosstab(df_predict['pessoa feliz'], df_predict['predict_arvore'], margins=True)
# %%
