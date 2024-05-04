import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

dados = pd.read_csv('C:\\Users\\Windows\\Downloads\\aerogerador.dat', sep='\s+', header=None, names=['velocidade', 'potencia'])
plt.scatter(dados['velocidade'], dados['potencia'])
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.title('Gráfico de Espalhamento: Velocidade do Vento vs Potência Gerada')
plt.show()

X = dados['velocidade'].values.reshape(-1, 1)
y = dados['potencia'].values.reshape(-1, 1)

num_rodadas = 1000

modelos = {
    'MQO Tradicional': LinearRegression(),
    'MQO Regularizado (Ridge)': Ridge(alpha=1.0),
    'Média de Valores Observados': None
}

lambda_values = [0.1, 1, 10, 100, 1000]
mse_medio_minimo = float('inf')
melhor_lambda = None

for l in lambda_values:
    modelo_ridge = Ridge(alpha=l)
    mse_medio = np.mean([mean_squared_error(modelo_ridge.fit(X_treino, y_treino).predict(X_teste), y_teste) 
                         for _ in range(num_rodadas)])
    if mse_medio < mse_medio_minimo:
        mse_medio_minimo = mse_medio
        melhor_lambda = l

print(f"O melhor valor de lambda encontrado foi: {melhor_lambda}")

eqm_resultados = {modelo: [] for modelo in modelos}

for _ in range(num_rodadas):
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2)
    y_pred_medio = np.mean(y_treino)
    
    for nome, modelo in modelos.items():
        if nome == 'Média de Valores Observados':
            eqm_resultados[nome].append(mean_squared_error(np.full_like(y_teste, y_pred_medio), y_teste))
        else:
            eqm_resultados[nome].append(mean_squared_error(modelo.fit(X_treino, y_treino).predict(X_teste), y_teste))

plt.figure(figsize=(10, 6))
for modelo, eqm in eqm_resultados.items():
    plt.hist(eqm, bins=30, alpha=0.5, label=modelo)

plt.xlabel('Erro Quadrático Médio')
plt.ylabel('Frequência')
plt.title('Distribuição do EQM para Modelos de Regressão')
plt.legend()
plt.show()
