from time import time as time
from matplotlib import pyplot as plt
import numpy as np

R = 100
lista_n = [1000, 10000, 100000]
t = []
for n in lista_n:
    tt = []
    for i in range(R):
        tic = time()
        for i in range(n):
            x =1
        toc = time()
        tt.append(toc-tic)
    t.append(tt)
    plt.plot(tt)
    tt.append(tt < toc)#filtrar tt
    np.mean(tt.filtrado)#media
    np.add(tt_filtrado)#desvio

    plt.title(f"n = {n}")
    plt.savefig(f"tempos_n{n}.png")



    
   # plt.show()

