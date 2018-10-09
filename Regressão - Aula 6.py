#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import pandas

#Mostra os itens da base
diabetes = load_diabetes()
diabetes.keys()
print(diabetes.DESCR)

tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names
tabela.head()


# In[3]:


tabela['Taxa'] = diabetes.target
tabela.head(10)


# In[4]:


tabela.corr()


# In[5]:


from sklearn.datasets import load_diabetes
import pandas

diabetes = load_diabetes()
tabela = pandas.DataFrame(diabetes.data)
tabela.columns = diabetes.feature_names

#seleciona duas colunas
X = tabela[["bmi", "s3"]]
print(X)


# In[11]:


#separa dados de treinamento do modelo linear e dados para validação do modelo
#inclui o modulo de regressão linear
from sklearn import linear_model
import matplotlib.pyplot as plt

tabela['Taxa'] = diabetes.target
#separa em dois conjuntos, um para treinamento e outro para validação (20 últimos)
X_t = X[:-20]
X_v = X[-20:]
#print(X_t["RM"])
y_t = tabela["Taxa"][:-20]
y_v = tabela["Taxa"][-20:]

regr = linear_model.LinearRegression()

# treina o modelo
regr.fit(X_t, y_t)

# faz a predição
y_pred = regr.predict(X_v)

# coeficientes a
print('Coeficientes: \n', regr.coef_)
#intercepto b
print('Coeficientes: \n', regr.intercept_)
#y = 814.255*bmi + -348.151*s3 + 152.8

#prediz manualmente os valores com base nos coeficientes encontrados na regressao
y_teste = 814.255*X_v["bmi"] - 348.151*X_v["s3"]+ 152.8
#print(y_teste)

#exibe o valor predito manualmente y_teste, que começa de 422
#exibe o valor real y_t
#exibe o valor predito pela regressão linear

print(y_teste[422], y_t[0],y_pred[0])

#plota todos os valores de validação
plt.scatter(X_v["s3"], y_v,  color='black')
plt.scatter(X_v["s3"], y_pred, color='blue')
plt.legend(["Real", "Predito"])


# In[ ]:




