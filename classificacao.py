#%%

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df

df['aprovado'] = (df['nota']>5).astype(int)
df
# %%

plt.plot(df['cerveja'], df['aprovado'], 'o', color='blue')
plt.grid(True)
plt.title('cerveja x aprovado')
plt.xlabel('cerveja')
plt.ylabel('aprovado')

#%%
from sklearn import linear_model
from sklearn import tree
from sklearn import naive_bayes

reg = linear_model.LogisticRegression(penalty=None, fit_intercept=True)

reg.fit(df[['cerveja']], df['aprovado'])
reg_predict = reg.predict(df[['cerveja']].drop_duplicates())
reg_proba = reg.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

arvore_full = tree.DecisionTreeClassifier(random_state=42)
arvore_full.fit(df[['cerveja']], df['aprovado'])
arvore_full_predict = arvore_full.predict(df[['cerveja']].drop_duplicates())
arvore_full_proba = arvore_full.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

arvore_d2 = tree.DecisionTreeClassifier(random_state=42, max_depth=2)
arvore_d2.fit(df[['cerveja']], df['aprovado'])
arvore_d2_predict = arvore_d2.predict(df[['cerveja']].drop_duplicates())
arvore_d2_proba = arvore_d2.predict_proba(df[['cerveja']].drop_duplicates())[:,1]

nb = naive_bayes.GaussianNB()
nb.fit(df[['cerveja']], df['aprovado'])
nb_predict = nb.predict(df[['cerveja']].drop_duplicates())
nb_proba = nb.predict_proba(df[['cerveja']].drop_duplicates())[:,1]



plt.plot(df['cerveja'], df['aprovado'], 'o', color='blue')
plt.grid(True)
plt.title('cerveja x aprovado')
plt.xlabel('cerveja')
plt.ylabel('aprovado')
plt.hlines(0.5, xmin=1, xmax=9, color='gray', linestyle='--')
plt.plot(df['cerveja'].drop_duplicates(), reg_predict, color='red')
plt.plot(df['cerveja'].drop_duplicates(), reg_proba, color='green')

plt.plot(df['cerveja'].drop_duplicates(), arvore_full_predict, color='orange')
plt.plot(df['cerveja'].drop_duplicates(), arvore_full_proba, color='purple')

plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_predict, color='brown')
plt.plot(df['cerveja'].drop_duplicates(), arvore_d2_proba, color='pink')

plt.plot(df['cerveja'].drop_duplicates(), nb_predict, color='cyan')
plt.plot(df['cerveja'].drop_duplicates(), nb_proba, color='magenta')

plt.legend(['observacao', 
            'reg predict', 
            'reg proba',
            'nb predict',
            'nb proba',
            'arvore full predict',
            'arvore full proba',
            #'arvore d2 predict',
            #'arvore d2 proba'
            ])
# %%
