from time import time
import matplotlib.pyplot as plt
import numpy as np

R = 100
lista_n = [1000, 10000, 100000]
t = []

for n in lista_n:
    tt = []
    for _ in range(R):
        tic = time()
        for i in range(n):
            x = 1
        toc = time()
        tt.append(toc - tic)
    
    # Converte para array NumPy para facilitar cálculos e filtragem
    tt_arr = np.array(tt)
    
    # Exemplo de filtragem: remove valores acima do percentil 95 (outliers)
    limite = np.percentile(tt_arr, 95)
    tt_filtrado = tt_arr[tt_arr <= limite]
    
    # Cálculo das estatísticas
    media = np.mean(tt_filtrado)
    desvio = np.std(tt_filtrado)
    
    # Geração do gráfico
    plt.figure()
    plt.plot(tt_arr, label="Todos os tempos")
    plt.axhline(media, color='r', linestyle='--', label=f"Média: {media:.6f}s")
    plt.title(f"Tempos de execução para n = {n}\nDesvio Padrão: {desvio:.6f}s")
    plt.xlabel("Repetição")
    plt.ylabel("Tempo (segundos)")
    plt.legend()
    plt.savefig(f"tempos_n{n}.png")
    plt.close() # Limpa a figura para a próxima iteração