import random
import math
import statistics
import numpy as np
import matplotlib.pyplot as plt

A = 10
dimensoes = 20
tamanho_populacao = 100
taxa_cruzamento = 0.7
taxa_mutacao = 0.01
num_geracoes = 100
bits_per_gene = 10

class Cromossomo:
    def __init__(self, bitstring):
        self.bitstring = bitstring
        self.genes = self.bitstring_to_genes()
        self.adaptacao = self.calcular_adaptacao()

    def bitstring_to_genes(self):
        genes = []
        for i in range(0, len(self.bitstring), bits_per_gene):
            gene_bitstring = self.bitstring[i:i + bits_per_gene]
            gene = int(gene_bitstring, 2) / (2**bits_per_gene - 1) * 20 - 10
            genes.append(gene)
        return genes

    def calcular_adaptacao(self):
        return A * dimensoes + sum(x**2 - A * math.cos(2 * math.pi * x) for x in self.genes) + 1

def inicializar_populacao():
    return [Cromossomo(''.join(random.choice(['0', '1']) for _ in range(dimensoes * bits_per_gene)))
            for _ in range(tamanho_populacao)]

def selecao_roleta(populacao):
    adaptacao_total = sum(cromossomo.adaptacao for cromossomo in populacao)
    probabilidades_selecao = [(cromossomo.adaptacao / adaptacao_total) for cromossomo in populacao]
    return populacao[np.random.choice(len(populacao), p=probabilidades_selecao)]

def cruzamento(pai1, pai2):
    if random.random() < taxa_cruzamento:
        ponto = random.randint(1, dimensoes * bits_per_gene - 1)
        bitstring_filho1 = pai1.bitstring[:ponto] + pai2.bitstring[ponto:]
        bitstring_filho2 = pai2.bitstring[:ponto] + pai1.bitstring[ponto:]
        return Cromossomo(bitstring_filho1), Cromossomo(bitstring_filho2)
    else:
        return pai1, pai2

def mutacao(cromossomo):
    bitlist = list(cromossomo.bitstring)
    for i in range(len(bitlist)):
        if random.random() < taxa_mutacao:
            bitlist[i] = '1' if bitlist[i] == '0' else '0'
    cromossomo.bitstring = ''.join(bitlist)
    cromossomo.genes = cromossomo.bitstring_to_genes()
    cromossomo.adaptacao = cromossomo.calcular_adaptacao()

def algoritmo_genetico():
    populacao = inicializar_populacao()
    aptidao_minima = []
    aptidao_maxima = []
    aptidao_media = []
    for _ in range(num_geracoes):
        nova_populacao = []
        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao_roleta(populacao)
            pai2 = selecao_roleta(populacao)
            filho1, filho2 = cruzamento(pai1, pai2)
            mutacao(filho1)
            mutacao(filho2)
            nova_populacao.extend([filho1, filho2])
        populacao = nova_populacao
        adaptacoes = [c.adaptacao for c in populacao]
        aptidao_minima.append(min(adaptacoes))
        aptidao_maxima.append(max(adaptacoes))
        aptidao_media.append(sum(adaptacoes) / len(adaptacoes))
    return aptidao_minima, aptidao_maxima, aptidao_media

aptidao_minima, aptidao_maxima, aptidao_media = algoritmo_genetico()

plt.figure(figsize=(10, 5))
plt.plot(aptidao_minima, label='Aptidão Minima')
plt.plot(aptidao_maxima, label='Aptidão Máxima')
plt.plot(aptidao_media, label='Aptidão Média')
plt.title('Aptidão por gerações')
plt.xlabel('Gerações')
plt.ylabel('Aptidão')
plt.legend()
plt.show()
