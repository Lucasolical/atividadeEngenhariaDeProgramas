from time import time
import numpy as np
from matplotlib import pyplot as plt

# Parâmetros
R = 100
lista_n = [1000, 10000, 100000]

desvios_padrao = []

print("Coletando dados e calculando o desvio padrão...")
for n in lista_n:
    tempos_n = []
    for r in range(R):
        tic = time()
        for i in range(n):
            x = 1
        toc = time()
        tempos_n.append(toc - tic)
    
    # Cálculo do desvio padrão absoluto para cada tamanho de entrada n
    desvios_padrao.append(np.std(tempos_n, ddof=1)) # ddof=1 para desvio padrão amostral

# Configuração e Plotagem do Gráfico
plt.figure(figsize=(10, 6))

plt.plot(
    lista_n, 
    desvios_padrao, 
    marker='s', 
    linestyle='-', 
    color='crimson', 
    linewidth=2.5, 
    markersize=8,
    label='Desvio Padrão Amostral'
)

# Ajuste de eixos para escala logarítmica para acompanhar a ordem de grandeza de n
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Tamanho da Entrada ($n$) [Escala Log]', fontsize=12)
plt.ylabel('Desvio Padrão do Tempo ($s$) [Escala Log]', fontsize=12)
plt.title('Gráfico de Desvio Padrão x Tamanho da Entrada ($n$)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Salvando a figura
plt.savefig("desvio_padrao_vs_n.png", dpi=300, bbox_inches='tight')
plt.show()