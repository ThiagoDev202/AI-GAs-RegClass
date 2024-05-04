import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

qnt_individuos = 50
qnt_max_geracoes = 100
valor_otimo_aceitavel = 10
Numero_elites = 5
numero_execucoes = 100

def inicializar_populacao(tamanho, numero_pontos):
    return np.array([np.random.permutation(numero_pontos) for _ in range(tamanho)], dtype=object)

def funcao_aptidao(cromossomo, pontos):
    soma_distancias = 0
    for i in range(len(cromossomo)):
        ponto_atual = pontos[cromossomo[i]]
        ponto_proximo = pontos[cromossomo[(i + 1) % len(cromossomo)]]
        distancia = np.linalg.norm(ponto_atual - ponto_proximo)
        soma_distancias += distancia
    return soma_distancias

def torneio(populacao, aptidoes, tamanho_torneio=3):
    selecionados = []
    while len(selecionados) < len(populacao):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        melhor = competidores[np.argmin(aptidoes[competidores])]
        selecionados.append(populacao[melhor])
    return np.array(selecionados, dtype=object)

def crossover_dois_pontos(pai1, pai2):
    tamanho = len(pai1)
    filho1, filho2 = np.zeros(tamanho, dtype=int), np.zeros(tamanho, dtype=int)
    ponto1, ponto2 = sorted(np.random.choice(range(tamanho), 2, replace=False))
    filho1[ponto1:ponto2+1], filho2[ponto1:ponto2+1] = pai2[ponto1:ponto2+1], pai1[ponto1:ponto2+1]
    def preencher_filho(filho, pai_original):
        pos = (ponto2 + 1) % tamanho
        indices_usados = set(filho[ponto1:ponto2+1])
        for gene in np.concatenate((pai_original[ponto2+1:], pai_original[:ponto2+1])):
            if gene not in indices_usados:
                filho[pos] = gene
                indices_usados.add(gene)
                pos = (pos + 1) % tamanho
    preencher_filho(filho1, pai1)
    preencher_filho(filho2, pai2)
    return filho1, filho2

def mutacao_por_troca(cromossomo, taxa_de_mutacao=0.01):
    for i in range(len(cromossomo)):
        if np.random.rand() < taxa_de_mutacao:
            j = np.random.randint(len(cromossomo))
            cromossomo[i], cromossomo[j] = cromossomo[j], cromossomo[i]
    return cromossomo

def elitismo(populacao, aptidoes, n_elites):
    elite_indices = np.argsort(aptidoes)[:n_elites]
    elites = populacao[elite_indices]
    return elites

def algoritmo_genetico():
    pontos = np.random.rand(10, 2)  
    populacao = inicializar_populacao(qnt_individuos, 10)
    populacao = np.array(populacao, dtype=object) 
    melhor_solucao = np.inf

    for _ in range(qnt_max_geracoes):
        aptidoes = np.array([funcao_aptidao(ind, pontos) for ind in populacao])
        if np.min(aptidoes) < melhor_solucao:
            melhor_solucao = np.min(aptidoes)
            if melhor_solucao <= valor_otimo_aceitavel:
                break  
        elites = elitismo(populacao, aptidoes, Numero_elites)
        selected = torneio(populacao, aptidoes)
        offspring = [crossover_dois_pontos(selected[i], selected[(i + 1) % len(selected)]) for i in range(0, len(selected), 2)]
        populacao = [mutacao_por_troca(child) for pair in offspring for child in pair]
        populacao.extend(elites)
        populacao = np.array(populacao, dtype=object)

    return melhor_solucao


resultados_aptidao = [algoritmo_genetico() for _ in range(numero_execucoes)]

menor_valor = np.min(resultados_aptidao)
maior_valor = np.max(resultados_aptidao)
media_valor = np.mean(resultados_aptidao)
desvio_padrao = np.std(resultados_aptidao)

tabela_resultados = pd.DataFrame({
    "Menor Valor de Aptidão": [menor_valor],
    "Maior Valor de Aptidão": [maior_valor],
    "Média de Valor de Aptidão": [media_valor],
    "Desvio-Padrão de Valor de Aptidão": [desvio_padrao]
})

print(tabela_resultados)

pontos = np.random.rand(10, 2) * 50
rotas = np.random.permutation(10)

def plot_tsp(pontos, rotas):
    plt.figure(figsize=(10, 6))
    x, y = pontos[:, 0], pontos[:, 1]
    
    plt.scatter(x, y, color='blue', s=100, zorder=5)
    plt.scatter(x[0], y[0], color='red', s=200, zorder=5, label='Origem') 
    
    for i in range(len(rotas)):
        start_pos = pontos[rotas[i]]
        end_pos = pontos[rotas[(i + 1) % len(rotas)]]
        plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', zorder=1)
    
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_tsp(pontos, rotas)