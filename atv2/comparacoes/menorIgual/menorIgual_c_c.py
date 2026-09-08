
from time import time
import matplotlib.pyplot as plt
import numpy as np

r=30
n=100000
tau_A = []

for r in range(r):
    t = 0
    for i in range(n):
        tic = time()
        1<=2
        toc = time()
        t+=toc - tic
    tau_A.append(t/n)

#controle
media = np.mean(tau_A)
Sigma = np.std(tau_A)
cv = Sigma/media

# if cv < 0.15 : ok!

print(f"Média do tempo por operação: {media:.10f} s")
print(f"Desvio Padrão (Sigma): {Sigma:.10f}")
print(f"Coeficiente de Variação (CV): {cv:.4f}")