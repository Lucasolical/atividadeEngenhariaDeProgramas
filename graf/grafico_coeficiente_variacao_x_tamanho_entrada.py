from time import time
import numpy as np
from matplotlib import pyplot as plt

# Parâmetros
R = 100
lista_n = [1000, 10000, 100000]

coef_variacao = []

print("Coletando dados e calculando o Coeficiente de Variação (CV)...")
for n in lista_n:
    tempos_n = []
    for r in range(R):
        tic = time()
        for i in range(n):
            x = 1
        toc = time()
        tempos_n.append(toc - tic)
    
    # Cálculo das estatísticas
    media = np.mean(tempos_n)
    desvio = np.std(tempos_n, ddof=1) # Desvio padrão amostral
    
    # Coeficiente de Variação (CV) = (Desvio Padrão / Média)
    # Proteção contra divisão por zero caso a média seja muito próxima de zero
    cv = (desvio / media) if media > 0 else 0.0
    coef_variacao.append(cv)

# Configuração e Plotagem do Gráfico
plt.figure(figsize=(10, 6))

plt.plot(
    lista_n, 
    coef_variacao, 
    marker='^', 
    linestyle='-', 
    color='forestgreen', 
    linewidth=2.5, 
    markersize=8,
    label='Coeficiente de Variação (CV)'
)

# Escala logarítmica apenas no eixo X para acomodar os diferentes tamanhos de n
# O eixo Y (CV) é mantido linear para evidenciar a porcentagem/proporção da dispersão relativa
plt.xscale('log')

plt.xlabel('Tamanho da Entrada ($n$) [Escala Log]', fontsize=12)
plt.ylabel('Coeficiente de Variação (CV = $\\sigma / \\mu$)', fontsize=12)
plt.title('Gráfico do Coeficiente de Variação x Tamanho da Entrada ($n$)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Salvando a figura
plt.savefig("coeficiente_variacao_vs_n.png", dpi=300, bbox_inches='tight')
plt.show()