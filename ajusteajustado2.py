from time import time
import matplotlib.pyplot as plt
import numpy as np

R = 100
lista_n = [1000, 10000, 100000]

# Fator multiplicador do desvio padrão (ex: 2 para ~95.4% ou 3 para ~99.7%)
x = 3

# Dicionário para guardar o tempo unitário de cada n
tempos_medios_unitarios = {}

for n in lista_n:
    tt = []
    for _ in range(R):
        tic = time()
        for i in range(n):
            x_val = 1  # renomeado para não conflitar com a variável x do multiplicador
        toc = time()
        tt.append(toc - tic)
    
    tt_arr = np.array(tt)
    
    # -------------------------------------------------------------
    # SUBSTITUIÇÃO DO PERCENTIL PELA FÓRMULA: t_filtrado < t + x*sigma
    # -------------------------------------------------------------
    media_inicial = np.mean(tt_arr)
    desvio_inicial = np.std(tt_arr)
    limite = media_inicial + (x * desvio_inicial)
    
    tt_filtrado = tt_arr[tt_arr < limite]
    # -------------------------------------------------------------
    
    # 1. Média e desvio padrão recalculados após a filtragem
    media_total = np.mean(tt_filtrado)
    desvio = np.std(tt_filtrado)
    coef_var = desvio / media_total
    
    # 2. Média do tempo de UMA ÚNICA atribuição
    tempo_unitario = media_total / n
    tempos_medios_unitarios[n] = tempo_unitario
    
    # Exibe no console para acompanhar
    print(f"n = {n}:")
    print(f"  - Tempo médio do laço completo: {media_total:.8f} s")
    print(f"  - Desvio padrão: {desvio:.8f} s")
    print(f"  - Tempo médio por atribuição: {tempo_unitario:.10f} s ({tempo_unitario * 1e9:.2f} ns)")
    print(f"  - Coeficiente de variação: {coef_var:.4f}\n")

    # Geração do gráfico
    plt.figure()
    plt.plot(tt_arr, label="Todos os tempos")
    plt.axhline(limite, color='g', linestyle=':', label=f"Limite superior ({x}σ): {limite:.6f}s")
    plt.axhline(media_total, color='r', linestyle='--', label=f"Média filtrada: {media_total:.6f}s")
    plt.title(f"Tempos de execução para n = {n}\nTempo unitário: {tempo_unitario*1e9:.2f} ns")
    plt.xlabel("Repetição")
    plt.ylabel("Tempo (segundos)")
    plt.legend()
    plt.savefig(f"tempos_n{n}.png")
    plt.close()

# Exibe o resumo final das 3 séries
print("--- RESUMO FINAL (Tempo por atribuição) ---")
for n, t_unit in tempos_medios_unitarios.items():
    print(f"n = {n:6d} -> {t_unit * 1e9:.2f} nanosegundos por 'x = 1'")