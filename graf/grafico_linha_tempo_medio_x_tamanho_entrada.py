from time import time
import numpy as np
from matplotlib import pyplot as plt

# Parâmetros
R = 100
lista_n = [1000, 10000, 100000]

tempos_medios = []
tempos_desvio = []

print("Coletando dados e calculando estatísticas para o gráfico de linha...")
for n in lista_n:
    tempos_n = []
    for r in range(R):
        tic = time()
        for i in range(n):
            x = 1
        toc = time()
        tempos_n.append(toc - tic)
    
    # Cálculo da média e desvio padrão para cada tamanho de entrada n
    tempos_medios.append(np.mean(tempos_n))
    tempos_desvio.append(np.std(tempos_n))

# Configuração e Plotagem do Gráfico de Linha
plt.figure(figsize=(10, 6))

# Gráfico de linha com marcadores e barras de erro (indicando a variabilidade/ruído)
plt.errorbar(
    lista_n, 
    tempos_medios, 
    yerr=tempos_desvio, 
    fmt='-o', 
    color='royalblue', 
    ecolor='crimson', 
    elinewidth=1.5, 
    capsize=4, 
    linewidth=2.5, 
    label='Tempo Médio $\pm$ Desvio Padrão'
)

# Ajuste de eixos para escala logarítmica (essencial para visualizar múltiplas ordens de grandeza)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Tamanho da Entrada ($n$) [Escala Log]', fontsize=12)
plt.ylabel('Tempo Médio de Execução ($s$) [Escala Log]', fontsize=12)
plt.title('Gráfico de Linha: Tempo Médio x Tamanho da Entrada ($n$)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Salvando a figura
plt.savefig("linha_tempo_medio_vs_n.png", dpi=300, bbox_inches='tight')
plt.show()