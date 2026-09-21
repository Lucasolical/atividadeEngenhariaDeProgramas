import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('atv4/imagem.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

n,m,c = img.shape
print(f"linhas: {n} colunas: {m} cases: {c}")
plano = []
faixa = []
for cor in range(c):
    plano.append(img[:,:,cor])
   #plt.imshow(plano[-1], cmap = 'gray')
    plt.show()
    faixa.append(plano[-1].flatten())
    plt.scatter(np.arange(len(faixa[-1])), faixa[-1])
    #plt.show()

k = 2
centroid = []
n_pixels = len(faixa[0])
for v in range(k):
    px = np.random.randint(0, n_pixels)
    v0 = faixa[0][px]
    v1 = faixa[1][px]
    v2 = faixa[2][px]
    centroid.append([v0,v1,v2])
    print(centroid[-1])

