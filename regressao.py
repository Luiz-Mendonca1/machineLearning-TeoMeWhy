#%%

import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")
df.head()
# %%

from sklearn import linear_model

X = df[['cerveja']] #matriz dataframe
y = df['nota'] #vetor series

reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(X, y)

# %%

a, b = reg.intercept_, reg.coef_[0] 
print(a, b)

#%%

predict = reg.predict(X.drop_duplicates())#pega unicos

#%%

import matplotlib.pyplot as plt

plt.plot(X['cerveja'], y, 'o')
plt.grid(True)
plt.title('Relacao cerveja nota')
plt.xlabel('cerveja')
plt.ylabel('nota')

plt.plot(X.drop_duplicates()['cerveja'], predict)

plt.legend(['observado', f'y = {a:.2f} + {b:.2f}*x'])
# %%
