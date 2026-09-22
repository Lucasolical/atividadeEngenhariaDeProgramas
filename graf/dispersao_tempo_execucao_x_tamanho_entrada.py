from time import time
import numpy as np
from matplotlib import pyplot as plt

# Parâmetros
R = 100
lista_n = [1000, 10000, 100000]

# Listas para armazenar todos os pontos (n, tempo)
x_vals = []
y_vals = []

print("Coletando dados de execução...")
for n in lista_n:
    for r in range(R):
        tic = time()
        for i in range(n):
            x = 1
        toc = time()
        tempo = toc - tic
        
        x_vals.append(n)
        y_vals.append(tempo)

# Convertendo para arrays do NumPy
x_arr = np.array(x_vals)
y_arr = np.array(y_vals)

# Tratamento e Separação de Outliers usando o método IQR para cada valor de n
x_norm, y_norm = [], []
x_out, y_out = [], []

for n in lista_n:
    # Filtra os tempos apenas para o n atual
    mask = (x_arr == n)
    tempos_n = y_arr[mask]
    
    # Cálculo dos quartis (Q1 e Q3) e do IQR
    q1 = np.percentile(tempos_n, 25)
    q3 = np.percentile(tempos_n, 75)
    iqr = q3 - q1
    
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    
    # Classifica cada execução entre normal ou outlier
    for t_val in tempos_n:
        if limite_inferior <= t_val <= limite_superior:
            x_norm.append(n)
            y_norm.append(t_val)
        else:
            x_out.append(n)
            y_out.append(t_val)

# Configuração e Plotagem do Gráfico de Dispersão
plt.figure(figsize=(10, 6))

# Pontos normais (com transparência para visualizar densidade/espalhamento)
plt.scatter(x_norm, y_norm, alpha=0.6, color='royalblue', label='Execuções Padrão', s=30)

# Outliers destacados (ruído do Sistema Operacional / Cache Misses)
if x_out:
    plt.scatter(x_out, y_out, alpha=0.9, color='crimson', marker='x', s=60, linewidth=2, label='Outliers (Ruído SO/Cache)')

# Ajustes de eixos em escala logarítmica para melhor visualização de múltiplas ordens de grandeza
plt.xscale('log')
plt.yscale('log')

plt.xlabel('Tamanho da Entrada ($n$) [Escala Log]', fontsize=12)
plt.ylabel('Tempo de Execução ($s$) [Escala Log]', fontsize=12)
plt.title('Gráfico de Dispersão: Tempo de Execução x Tamanho da Entrada', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Salvando a figura
plt.savefig("dispersao_tempo_vs_n.png", dpi=300, bbox_inches='tight')
plt.show()