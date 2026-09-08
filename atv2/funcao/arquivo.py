from time import time
import numpy as np

# Função a ser medida
def minha_funcao():
    pass 

def funcao_com_calculo(a, b):
    return a + b

r = 30
n = 100000
tau_A = []

for _ in range(r):
    tic = time()
    for _ in range(n):
        funcao_com_calculo(1,2)
    toc = time()
    
    tempo_medio = (toc - tic) / n
    tau_A.append(tempo_medio)

media = np.mean(tau_A)
Sigma = np.std(tau_A)
cv = Sigma / media

# Resultados
print(f"Média do tempo por operação: {media:.10f} s ({media * 1e9:.2f} ns)")
print(f"Desvio Padrão (Sigma): {Sigma:.10f}")
print(f"Coeficiente de Variação (CV): {cv:.4f}")

if cv < 0.15:
    print("Resultado OK! (CV < 0.15)")
else:
    print("Aviso: Variação alta (CV >= 0.15).")