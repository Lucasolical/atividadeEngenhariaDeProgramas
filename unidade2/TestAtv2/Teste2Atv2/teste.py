import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
import gc
import random
from pathlib import Path

# Diretório onde está o arquivo teste.py
base_dir = Path(__file__).resolve().parent

# Sobe um nível: Teste2Atv2 -> TestAtv2
projeto_dir = base_dir.parent

# Caminho correto para as imagens
img_dir = projeto_dir / "minhas_imagens"

# os.sched_setaffinity(0, {13})

def fcm_processar_imagem(img, K, m=2.0, max_iter=100, epsilon=0.01):
    """
    Executa o pipeline completo do FCM Puro com proteções contra índice inválido e divisão por zero.
    """
    
    largura, altura = img.size
    
    # Extração segura dos pixels da imagem
    pixels_2d = list(img.getdata())
    
    # ====================================================================
    # ALVO: MISSÃO 3 (Structure of Arrays - SoA)
    # ====================================================================
    X = []
    for p in pixels_2d:
        X.append([float(p[0]), float(p[1]), float(p[2])])
        
    N = len(X)
    canais = 3
    
    paleta_cores = [
        [255, 0, 0],     # Cluster 0: Vermelho
        [0, 255, 0],     # Cluster 1: Verde
        [0, 0, 255],     # Cluster 2: Azul
        [255, 255, 0],   # Cluster 3: Amarelo
        [255, 0, 255],   # Cluster 4: Magenta
        [0, 255, 255]    # Cluster 5: Ciano
    ]
    
    # ====================================================================
    # ALVO: MISSÃO 1 (Memory Flattening)
    # ====================================================================
    U = []
    for i in range(N):
        linha = []
        soma_linha = 0.0
        for j in range(K):
            val = random.random()
            linha.append(val)
            soma_linha += val
        
        for j in range(K):
            if soma_linha == 0.0:
                linha[j] = 1.0 / K
            else:
                linha[j] = linha[j] / soma_linha
        U.append(linha)
        
    # 4. LAÇO PRINCIPAL DO FCM (Puro)
    for iteracao in range(max_iter):
        
        # ====================================================================
        # ALVO: MISSÃO 2 (Fixed Point Math / Fuga da FPU)
        # ====================================================================
        C = []
        for j in range(K):
            numerador = [0.0] * canais
            denominador = 0.0
            
            for i in range(N):
                u_m = U[i][j] ** m
                denominador += u_m
                
                # ====================================================================
                # ALVO: MISSÃO 4 (Loop Unrolling)
                # ====================================================================
                for c in range(canais):
                    numerador[c] += u_m * X[i][c]
            
            centroide_j = []
            for c in range(canais):
                if denominador == 0.0:
                    centroide_j.append(0.0)
                else:
                    centroide_j.append(numerador[c] / denominador)
            C.append(centroide_j)
            
        U_nova = []
        max_diff = 0.0
        
        # ====================================================================
        # ALVO: MISSÃO 6 (Loop Fission)
        # ====================================================================
        for i in range(N):
            linha_nova = [0.0] * K
            distancias = [0.0] * K
            
            for j in range(K):
                soma_quadrados = 0.0
                for c in range(canais):
                    diff = X[i][c] - C[j][c]
                    soma_quadrados += diff * diff
                distancias[j] = soma_quadrados ** 0.5 
                
                # ====================================================================
                # ALVO: MISSÃO 5 (Early Exit / Poda Condicional)
                # ====================================================================
                
            for j in range(K):
                if distancias[j] == 0.0:
                    linha_nova[j] = 1.0
                    for k in range(K):
                        if k != j: 
                            linha_nova[k] = 0.0
                    break
                    
                soma_dist = 0.0
                for k in range(K):
                    d_k = distancias[k] if distancias[k] > 0.0 else 1e-10
                    razao = distancias[j] / d_k
                    soma_dist += razao ** (2.0 / (m - 1.0))
                
                pertinencia = 1.0 / soma_dist if soma_dist > 0.0 else (1.0 / K)
                linha_nova[j] = pertinencia
                
                diff_u = pertinencia - U[i][j]
                if diff_u < 0: 
                    diff_u = -diff_u
                if diff_u > max_diff: 
                    max_diff = diff_u
                
            U_nova.append(linha_nova)
            
        # O U TEM QUE SER ATUALIZADO AQUI FORA DO LAÇO DE PIXELS!
        U = U_nova
            
        if max_diff < epsilon:
            print(f"-> Convergiu na iteração {iteracao}!")
            break
            
    # 5. FASE 5: COLORIZAÇÃO POR PERTINÊNCIA MÁXIMA
    pixels_segmentados_lista = []
    for i in range(N):
        maior_pertinencia = -1.0
        cluster_vencedor = 0
        for j in range(K):
            if U[i][j] > maior_pertinencia:
                maior_pertinencia = U[i][j]
                cluster_vencedor = j
                
        cor = paleta_cores[cluster_vencedor % len(paleta_cores)]
        pixels_segmentados_lista.append((cor[0], cor[1], cor[2]))
        
    # 6. RESHAPE MANUAL E SALVAMENTO
    imagem_final = Image.new("RGB", (largura, altura))
    imagem_final.putdata(pixels_segmentados_lista)
    
    return C, U, imagem_final

# ==========================================
# 1. Função do Aluno (FCM Puro)
# ==========================================
def segmenta_imagem(imagem_array, k):
    """
    Substitua pelo script real de segmentação do aluno.
    Recebe o array da imagem já em memória e o número de clusters K.
    """
    # Converte o array numpy da imagem para o formato PIL que o FCM Puro exige
    img_pil = Image.fromarray(imagem_array.astype('uint8'), 'RGB')
    
    # Chama a função do FCM Puro
    centroides, pertinencias, imagem_segmentada = fcm_processar_imagem(img_pil, K=k)
    
def exibir_equacao_ajustada(popt, X_data, y_data):
    """
    Exibe no terminal a equação ajustada com notação científica legível,
    os coeficientes individuais e o coeficiente de determinação (R²).
    """
    alpha, beta, gamma, delta = popt
    
    # Cálculo do R² (Coeficiente de Determinação)
    y_pred = model_n_k(X_data, *popt)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    print("\n" + "="*60)
    print("   MODELO EMPÍRICO AJUSTADO: T(n, K)")
    print("="*60)
    print("Formulação Geral:")
    print("  T(n, K) = α·(n·K) + β·n + γ·K + δ\n")
    print("Parâmetros Identificados:")
    print(f"  α (alpha) = {alpha:14.6e} s/(pixel·cluster)")
    print(f"  β (beta)  = {beta:14.6e} s/pixel")
    print(f"  γ (gamma) = {gamma:14.6e} s/cluster")
    print(f"  δ (delta) = {delta:14.6e} s (overhead base)")
    print("-" * 60)
    print("Equação Parametrizada:")
    print(f"  T(n, K) = ({alpha:.4e})·(n·K) + ({beta:.4e})·n + ({gamma:.4e})·K + ({delta:.4e})")
    print(f"Aderência Estatística (R²): {r2:.4f}")
    print("="*60 + "\n")


# ==========================================
# 2. Descoberta e Ordenação Ligeira (Sem estourar RAM)
# ==========================================
# k_list = [2, 3, 4]
k_list = [2,3]
# R = 30
R = 10 # Número de repetições por combinação

print("Mapeando imagens e calculando 'n' (apenas cabeçalhos)...")
arquivos_n = []

# Varre os arquivos no diretório especificado
if os.path.exists(img_dir):
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', 'avif')):
            filepath = os.path.join(img_dir, filename)
            
            # Image.open lê apenas os metadados. Não aloca a matriz de pixels.
            with Image.open(filepath) as img:
                w, h = img.size
                n = w * h
                arquivos_n.append({'arquivo': filename, 'caminho': filepath, 'n': n})

# Ordena a lista de dicionários pelo valor de 'n' crescente
arquivos_n = sorted(arquivos_n, key=lambda x: x['n'])

print(f"{len(arquivos_n)} imagens ordenadas prontas para processamento.")

# ==========================================
# 3. Execução (Carregando sob demanda)
# ==========================================
resultados = []

for item in arquivos_n:
    n = item['n']
    filepath = item['caminho']
    nome_arquivo = item['arquivo']
    
    print(f"Carregando {nome_arquivo} para a RAM (n={n})...")
    
    # Carrega a imagem real para a RAM apenas no momento de usar
    with Image.open(filepath) as img:
        img_array = np.array(img)
    
    # Executa todas as repetições e Ks para esta imagem
    for k in k_list:
        
        gc.collect()
        gc.disable()
        
        try:
                
            for r in range(R):
                t0 = time.perf_counter()
                segmenta_imagem(img_array, k)  # A matriz é passada aqui
                t1 = time.perf_counter()
                
                resultados.append({
                    'arquivo': nome_arquivo,
                    'n': n,     
                    'k': k, 
                    'repeticao': r, 
                    'tempo': (t1 - t0)
                })

        finally:
            gc.enable()   
            
    # Libera a RAM explicitamente antes de carregar a próxima imagem
    del img_array
    print(f"Análise de {nome_arquivo} concluída. Memória liberada.")

# Transforma os resultados em DataFrame
df_raw = pd.DataFrame(resultados)

# ==========================================
# 4. Pré-processamento e Validação F.O.C.A.
# ==========================================
if not df_raw.empty:
    df_filtered = pd.DataFrame()
    
    # Filtragem de Outliers (Percentis 5 e 95)
    for (n, k), group in df_raw.groupby(['n', 'k']):
        p05 = group['tempo'].quantile(0.05)
        p95 = group['tempo'].quantile(0.95)
        valid_mask = (group['tempo'] >= p05) & (group['tempo'] <= p95)
        df_filtered = pd.concat([df_filtered, group[valid_mask]])
    
    # Validação de Estabilidade (Média, Desvio e CV)
    resumo = df_filtered.groupby(['n', 'k']).agg(
        tempo_medio=('tempo', 'mean'),
        desvio=('tempo', 'std'),
        contagem_valida=('tempo', 'count')
    ).reset_index()
    
    resumo['CV'] = resumo['desvio'] / resumo['tempo_medio']
    resumo['estavel_CV_menor_15'] = resumo['CV'] < 0.15
    
    print("\n--- Resumo de Estabilidade ---")
    print(resumo)
    
    # ==========================================
    # 5. Ajuste de Curvas
    # ==========================================
    def model_n_k(X, alpha, beta, gamma, delta):
        n, k = X
        return alpha * (n * k) + beta * n + gamma * k + delta
    
    X_data = np.vstack((resumo['n'], resumo['k']))
    y_data = resumo['tempo_medio']
    
    # O p0 pode precisar de sintonia dependendo da escala de tempo real
    popt, _ = curve_fit(model_n_k, X_data, y_data, p0=[1e-6, 1e-6, 1e-6, 0.01], maxfev=10000)
    resumo['tempo_predito'] = model_n_k(X_data, *popt)
    exibir_equacao_ajustada(popt, X_data, y_data)
    
    # ==========================================
    # 6. Plotagens Bidimensionais
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot A: Tempo vs n
    colors_k = ['red', 'green', 'blue', 'orange', 'purple'] 
    for idx, k in enumerate(k_list):
        dados_k = resumo[resumo['k'] == k]
        if not dados_k.empty:
            c = colors_k[idx % len(colors_k)]
            ax1.plot(dados_k['n'], dados_k['tempo_medio'], marker='o', 
                     color=c, label=f'K={k} (Empírico)')
            ax1.plot(dados_k['n'], dados_k['tempo_predito'], linestyle='--', 
                     color=c, alpha=0.6)
    ax1.set_xlabel('Número de Pixels (n)')
    ax1.set_ylabel('Tempo Médio de Execução (s)')
    ax1.set_title('Tempo vs Tamanho da Imagem')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot B: Tempo vs K
    n_processados = sorted(resumo['n'].unique())
    colors_n = plt.cm.viridis(np.linspace(0, 1, len(n_processados)))
    
    for idx, n_val in enumerate(n_processados):
        dados_n = resumo[resumo['n'] == n_val]
        if not dados_n.empty:
            ax2.plot(dados_n['k'], dados_n['tempo_medio'], marker='s', 
                     color=colors_n[idx], label=f'n={n_val}')
            ax2.plot(dados_n['k'], dados_n['tempo_predito'], linestyle='--', 
                     color=colors_n[idx], alpha=0.6)
    
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_title('Tempo vs Número de Clusters')
    ax2.set_xticks(k_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../outputs/relatorio_complexidade.png', dpi=300)
    plt.show()

     # ==========================================
    # 6.5. Plotagem da Superfície 3D
    # Tempo(n, k)
    # ==========================================

    fig_3d = plt.figure(figsize=(10, 8))

    ax_3d = fig_3d.add_subplot(
        111,
        projection='3d'
    )

    # Cria uma malha de valores para n e k
    n_linspace = np.linspace(
        resumo['n'].min(),
        resumo['n'].max(),
        50
    )

    k_linspace = np.linspace(
        resumo['k'].min(),
        resumo['k'].max(),
        50
    )

    N_mesh, K_mesh = np.meshgrid(
        n_linspace,
        k_linspace
    )

    # Calcula os tempos previstos pelo modelo ajustado
    T_mesh = model_n_k(
        (N_mesh, K_mesh),
        *popt
    )

    # Desenha a superfície teórica
    surf = ax_3d.plot_surface(
        N_mesh,
        K_mesh,
        T_mesh,
        cmap='viridis',
        alpha=0.7,
        linewidth=0,
        antialiased=True
    )

    # Adiciona a barra de cores
    fig_3d.colorbar(
        surf,
        ax=ax_3d,
        shrink=0.6,
        aspect=12,
        pad=0.1,
        label='Tempo previsto (s)'
    )

    # Sobrepõe os dados experimentais
    ax_3d.scatter(
        resumo['n'],
        resumo['k'],
        resumo['tempo_medio'],
        color='red',
        edgecolor='black',
        s=60,
        label='Dados experimentais',
        depthshade=False
    )

    # Configura os eixos
    ax_3d.set_xlabel('Número de pixels (n)')
    ax_3d.set_ylabel('Número de clusters (k)')
    ax_3d.set_zlabel('Tempo de execução (s)')

    ax_3d.set_title(
        'Superfície de Tempo de Execução T(n, k)'
    )

    # Ajusta o ângulo de visualização
    ax_3d.view_init(
        elev=25,
        azim=135
    )

    ax_3d.legend()

    plt.tight_layout()

    # Salva a figura
    plt.savefig(
        '../outputs/superficie_3d_foca.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.show()

    print(
        "\nSuperfície 3D gerada e salva em "
        "'../outputs/superficie_3d_foca.png'."
    )

    print("\nPipeline concluído. Gráficos gerados e salvos como 'relatorio_complexidade.png'.")
    
    # ==========================================
    # 7. Exportação dos Dados para Avaliação
    # ==========================================
    arquivo_csv = 'resultados_foca_aluno.csv'
    
    # Exporta os dados usando ponto e vírgula como separador e vírgula para decimais,
    # garantindo que o arquivo seja aberto corretamente no Excel em PT-BR.
    resumo.to_csv(arquivo_csv, index=False, float_format='%.6f', sep=';', decimal=',')
    
    print(f"Tabela de avaliação exportada com sucesso para: {arquivo_csv}")
    
else:
    print("\nNenhum dado processado. Verifique se o diretório contém imagens válidas.")