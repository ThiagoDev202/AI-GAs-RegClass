import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

dados_caixeiro_viajante = pd.read_csv('C:\\Users\\Windows\\Downloads\\CaixeroGrupos.csv', header=None)
pontos = dados_caixeiro_viajante.iloc[:, :3].values
grupos = dados_caixeiro_viajante.iloc[:, 3].values.astype(int)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

cores = ['r', 'g', 'b', 'y', 'm', 'c']
cor = [cores[i % len(cores)] for i in grupos]

sc = ax.scatter(pontos[:, 0], pontos[:, 1], pontos[:, 2], c=cor)

elementos = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cores[i % len(cores)], label=f'Group {i}', markersize=10)
                   for i in np.unique(grupos)]
ax.legend(handles=elementos, title='Grupos')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
plt.show()
