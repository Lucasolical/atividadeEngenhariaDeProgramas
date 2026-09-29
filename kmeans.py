# ============================================================
# K-MEANS PARA SEGMENTAÇÃO DE IMAGEM
#
# I = Inicialização
# A = Atribuição
# C = Cálculo/atualização dos centroides
# V = Verificação
#
# I → A → C → V → A → C → V → ...
# ============================================================


# ============================================================
# 1. IMPORTAÇÃO E LEITURA DA IMAGEM
# ============================================================

# Importamos as ferramentas necessárias:
# - matplotlib.pyplot: essencial para desenhar e mostrar os gráficos e imagens na tela.
# - cv2 (OpenCV): biblioteca usada para abrir, ler e manipular imagens digitais.
# - numpy: biblioteca fundamental para trabalhar com matrizes e cálculos matemáticos rápidos.
# - time: usada para marcar o tempo de execução do script do começo ao fim.
# - os: ajuda a interagir com o sistema operacional (como verificar a pasta atual).
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os

# Mostramos no terminal qual é a pasta atual onde o script está rodando
print("Diretório atual:", os.getcwd())

# Marcamos o tempo de início para sabermos quanto tempo o algoritmo demorará no total
inicio = time.time()

# Carregamos a imagem do disco usando o OpenCV (arquivo 'teste2.jpeg')
img = cv2.imread('teste2.jpeg')

# O OpenCV lê as imagens no formato BGR (Azul, Verde, Vermelho).
# Como o Python/matplotlib trabalha com RGB, transformamos a imagem para o padrão correto.
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Redimensionamos a imagem para um tamanho fixo (360 linhas por 800 colunas).
# Isso é essencial para padronizar a entrada e acelerar o processamento dos pixels.
img = cv2.resize(img, (360, 800))

# Criamos uma janela visual para mostrar a imagem original carregada
plt.imshow(img)
plt.show()


# ============================================================
# 2. DEFINIÇÕES INICIAIS
# ============================================================

# Definimos o número de grupos (clusters) que o K-Means vai procurar.
K = 2


# ============================================================
# I — INICIALIZAÇÃO
# ============================================================

# Descobrimos as dimensões da matriz da imagem:
# - n: quantidade de linhas (altura)
# - m: quantidade de colunas (largura)
# - c: quantidade de canais de cor (3 para o padrão RGB: Vermelho, Verde e Azul)
n, m, c = img.shape

print(f'Linhas = {n}')
print(f'Colunas = {m}')
print(f'Cores = {c}')


# ------------------------------------------------------------
# Separar os canais R, G e B
# ------------------------------------------------------------

# Criamos listas vazias para guardar os planos de cor separados e transformados
faixa = []
plano = []

# Um loop que passa por cada canal de cor (0 para Vermelho, 1 para Verde, 2 para Azul)
for cor in range(c):

    # Separamos apenas a matriz correspondente àquele canal de cor na imagem inteira
    plano.append(img[:, :, cor])

    # Transformamos a matriz bidimensional da cor em uma lista linear de pixels (vetor de 1 dimensão)
    # Isso facilita muito na hora de varrer todos os pixels individualmente.
    faixa.append(plano[-1].flatten())

    # Geramos um gráfico de dispersão para mostrar a distribuição dos valores daquele canal de cor
    plt.scatter(
        np.arange(len(faixa[-1])),
        faixa[-1],
        s=0.1
    )
    plt.show()


# ------------------------------------------------------------
# Número de pixels
# ------------------------------------------------------------

# Calculamos o número total de pixels da imagem multiplicando linhas por colunas (tamanho do vetor de cor)
n_pixels = len(faixa[0])

print(f'Número de pixels = {n_pixels}')


# ------------------------------------------------------------
# Escolher os centroides iniciais
# ------------------------------------------------------------

# Criamos uma lista vazia para guardar os centroides (as cores de referência iniciais de cada grupo)
centroides = []

# Repetimos o processo para cada um dos K grupos que criaremos
for i in range(K):

    # Sorteamos a posição de um pixel aleatório dentro do total de pixels da imagem
    px = np.random.randint(0, n_pixels)

    # Criamos uma lista vazia para armazenar os valores de cor (RGB) desse pixel sorteado
    v = []

    # Pegamos o valor correspondente àquele pixel em cada um dos canais de cor (R, G e B)
    for f in faixa:
        v.append(f[px])

    # Adicionamos esse vetor de cores como um centroide inicial oficial do algoritmo
    centroides.append(v)

    print(f'Centroide inicial [{i}] = {centroides[i]}')


# ============================================================
# A → C → V (O Coração do Algoritmo K-Means)
# ============================================================

# Contador para sabermos quantas iterações (voltas) o algoritmo vai dar até convergir
iteracao = 0

# Laço de repetição infinito que só vai parar quando o algoritmo estabilizar (convergir)
while True:

    iteracao += 1

    print()
    print('============================================================')
    print(f'ITERAÇÃO {iteracao}')
    print('============================================================')


    # ========================================================
    # A — ATRIBUIÇÃO (Associar cada pixel ao centroide mais próximo)
    # ========================================================

    # Criamos uma lista de listas para guardar quais pixels pertencem a cada cluster.
    # Exemplo: cluster[0] guardará os índices de todos os pixels que vão pertencer ao grupo 0.
    cluster = [[] for _ in range(K)]

    # Percorremos um por um todos os pixels da imagem inteira
    for px in range(n_pixels):

        # ----------------------------------------------------
        # Criar o vetor RGB do pixel atual
        # ----------------------------------------------------
        # Juntamos os valores de Vermelho, Verde e Azul do pixel que estamos analisando
        v = [
            faixa[0][px],   # R
            faixa[1][px],   # G
            faixa[2][px]    # B
        ]

        # ----------------------------------------------------
        # Calcular distância para cada centroide
        # ----------------------------------------------------
        # Lista temporária para guardar as distâncias entre este pixel e os centroides
        d = []

        for i in range(K):
            # Usamos a norma Euclidiana (fórmula de distância entre pontos) para medir
            # o quão parecida é a cor do pixel atual em relação à cor do centroide atual.
            distancia = np.linalg.norm(
                np.array(v) - np.array(centroides[i])
            )
            d.append(distancia)

        # ----------------------------------------------------
        # Encontrar o centroide mais próximo
        # ----------------------------------------------------
        # Identificamos qual centroide teve a menor distância em relação ao pixel.
        # O índice desse centroide vencedor define a qual cluster o pixel vai pertencer.
        cluster_id = np.argmin(d)

        # ----------------------------------------------------
        # Colocar o pixel no cluster correspondente
        # ----------------------------------------------------
        # Guardamos a posição (índice) do pixel dentro da lista do seu respectivo cluster.
        cluster[cluster_id].append(px)


    # --------------------------------------------------------
    # Mostrar resultado da atribuição
    # --------------------------------------------------------
    print()
    print('Quantidade de pixels em cada cluster:')

    for i in range(K):
        print(f'Cluster {i}: {len(cluster[i])} pixels')


    # ========================================================
    # C — Cálculo / Atualização dos Centroides (Recalcular as cores médias)
    # ========================================================

    # Lista temporária para guardar os novos centroides recalculados desta rodada
    novo_centroid = []

    for k in range(K):

        # ----------------------------------------------------
        # Quantidade de pixels no cluster
        # ----------------------------------------------------
        # Vemos quantos pixels caíram dentro deste cluster específico
        L = len(cluster[k])

        # ----------------------------------------------------
        # Soma R, G e B
        # ----------------------------------------------------
        # Criamos um vetor com zeros para acumular a soma das cores de todos os pixels do cluster
        S = np.zeros(3)

        for px in cluster[k]:
            S += [
                faixa[0][px],
                faixa[1][px],
                faixa[2][px]
            ]

        # ----------------------------------------------------
        # Média de R, G e B
        # ----------------------------------------------------
        # Calculamos a cor média exata dividindo a soma total das cores pela quantidade de pixels.
        # Essa média passa a ser a nova cor representativa (o novo centroide) daquele grupo.
        novo = [
            S[0] / L,
            S[1] / L,
            S[2] / L
        ]

        # Guardamos esse novo centroide na nossa lista de atualização
        novo_centroid.append(novo)


    print()
    print('Novos centroides:')

    for i in range(K):
        print(f'Centroide [{i}] = {novo_centroid[i]}')


    # ========================================================
    # V — VERIFICAÇÃO (Critério de Parada)
    # ========================================================

    # Comparamos matematicamente os centroides antigos com os novos centroides recém-calculados.
    # A função np.allclose verifica se eles são praticamente idênticos.
    if np.allclose(centroides, novo_centroid):

        print()
        print('K-Means convergiu!')
        print(f'Número de iterações: {iteracao}')

        # Se os centroides não mudaram mais, significa que o algoritmo achou o resultado ideal.
        # Interrompemos o loop infinito (while True).
        break

    else:

        print()
        print('Centroides mudaram.')
        print('Repetindo A → C → V...')

        # Se ainda mudaram, atualizamos os centroides antigos com os novos valores
        # e o algoritmo volta para o passo de Atribuição (A).
        centroides = novo_centroid


# ============================================================
# 3. SEGMENTAÇÃO DA IMAGEM (MÁSCARA BINÁRIA PRETO E BRANCO)
# ============================================================

# Criamos uma matriz de imagem vazia com o mesmo tamanho, altura e largura da imagem original
img_segmentada = np.zeros_like(img)

# Definimos as cores exatas exigidas pelo professor para a máscara binária (K = 2):
# - Cluster 0 vai receber a cor Preta [0, 0, 0]
# - Cluster 1 vai receber a cor Branca [255, 255, 255]
cores_binarias = [
    [0, 0, 0],       # Preto
    [255, 255, 255]  # Branco
]

# Passamos por cada grupo (cluster) criado
for k in range(K):
    # Passamos por cada pixel que pertence àquele grupo
    for px in cluster[k]:
        # Convertemos a posição linear do pixel de volta para coordenadas visuais (linha e coluna)
        linha = px // m
        coluna = px % m

        # Pintamos o pixel correspondente na imagem segmentada com a cor preta ou branca correspondente
        img_segmentada[linha, coluna] = cores_binarias[k]

# Convertemos o tipo dos dados da imagem segmentada para inteiros de 8 bits (padrão correto de imagem do OpenCV)
img_segmentada = img_segmentada.astype(np.uint8)


# ============================================================
# 4. MOSTRAR RESULTADO
# ============================================================

# Configuramos o tamanho da janela onde os gráficos serão exibidos lado a lado
plt.figure(figsize=(10, 5))

# Criamos o primeiro espaço na tela (subgráfico 1) para mostrar a imagem original
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Imagem Original')
plt.axis('off')  # Desliga as bordas e numeração dos eixos para ficar limpo

# Criamos o segundo espaço na tela (subgráfico 2) para mostrar a máscara binária gerada
plt.subplot(1, 2, 2)
plt.imshow(img_segmentada)
plt.title(f'Imagem Segmentada - K={K}')
plt.axis('off')  # Desliga as bordas e numeração dos eixos

# Mostramos a janela definitiva com as duas imagens lado a lado na tela
plt.show()


# ============================================================
# 5. TEMPO TOTAL
# ============================================================

# Capturamos o horário atual do fim do processamento
fim = time.time()

# Subtraímos o tempo inicial do tempo final para descobrir quantos segundos o script demorou
tempo = fim - inicio

print()
print(f'Tempo total: {tempo:.4f} segundos')


# ============================================================
# RESUMO DO ALGORITMO
#
# I → Inicialização
#
#     Escolhe K pixels aleatórios
#     para serem os centroides iniciais.
#
#
# A → Atribuição
#
#     Para cada pixel:
#         calcula a distância para cada centroide
#         encontra o centroide mais próximo
#         coloca o pixel naquele cluster
#
#
# C → Cálculo
#
#     Para cada cluster:
#         soma R, G e B dos seus pixels
#         calcula a média
#         cria um novo centroide
#
#
# V → Verificação
#
#     Compara os centroides antigos
#     com os novos centroides.
#
#     Se forem iguais:
#         FIM
#
#     Se forem diferentes:
#         centroides = novos centroides
#         volta para A
#
#
# I → A → C → V → A → C → V → ...
# ============================================================
