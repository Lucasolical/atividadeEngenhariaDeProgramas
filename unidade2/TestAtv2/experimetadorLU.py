import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit

# ==========================================
# 1. Função de Segmentação do Aluno (Integrada e Otimizada)
# ==========================================
def segmenta_imagem(imagem_array, k):
    """
    Algoritmo K-Means personalizado para segmentação de imagem.
    Recebe o array da imagem já em memória e o número de clusters K.
    """
    n_linhas, n_colunas, c = imagem_array.shape
    pixels = imagem_array.reshape(-1, c)
    n_pixels = pixels.shape[0]
    
    # 3. Inicialização aleatória dos K centroides
    centroid_indices = np.random.choice(n_pixels, k, replace=False)
    centroids_arr = pixels[centroid_indices].astype(float)
    
    # 4. Atribuição dos pixels ao centroide mais próximo (Vetorizado para eficiência)
    dists = np.linalg.norm(pixels[:, np.newaxis, :] - centroids_arr[np.newaxis, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    
    # 5. Re-cálculo dos novos centroides
    novo_centroid = []
    for l in range(k):
        cluster_pixels = pixels[labels == l]
        if len(cluster_pixels) > 0:
            media_rgb = cluster_pixels.mean(axis=0)
            novo_centroid.append(media_rgb)
        else:
            novo_centroid.append(centroids_arr[l])
            
    novo_centroid = np.array(novo_centroid)
    
    # Reconstrói a imagem segmentada substituindo cada pixel pelo seu centroide
    seg_pixels = novo_centroid[labels]
    seg_img = seg_pixels.reshape(n_linhas, n_colunas, c).astype(np.uint8)
    
    return seg_img

# ==========================================
# DEFINIÇÃO DAS FUNÇÕES MATEMÁTICAS E DE EXIBIÇÃO
# ==========================================
def model_n_k(X, alpha, beta, gamma, delta):
    n, k = X
    return alpha * (n * k) + beta * n + gamma * k + delta

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
# EXECUÇÃO PRINCIPAL
# ==========================================
if __name__ == "__main__":
    tic = time.time()

    img_dir = "./minhas_imagens" # Substitua pelo diretório real
    k_list = [2, 3, 4]
    
    # IMPORTANTE: Altere para R=30 quando for rodar o teste oficial final
    R = 30 

    print("Mapeando imagens e calculando 'n' (apenas cabeçalhos)...")
    arquivos_n = []

    # Varre os arquivos no diretório especificado
    if os.path.exists(img_dir):
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
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
        X_data = np.vstack((resumo['n'], resumo['k']))
        y_data = resumo['tempo_medio']
        
        try:
            # O p0 pode precisar de sintonia dependendo da escala de tempo real
            popt, _ = curve_fit(model_n_k, X_data, y_data, p0=[1e-6, 1e-6, 1e-6, 0.01], maxfev=10000)
            resumo['tempo_predito'] = model_n_k(X_data, *popt)
            
            # Chamada da função corrigida
            exibir_equacao_ajustada(popt, X_data, y_data)
            
            # ==========================================
            # 6. Plotagens Bidimensionais
            # ==========================================
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
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
            plt.savefig('relatorio_complexidade_kmeans.png', dpi=300)
            # plt.show()  # Mantido comentado para não travar o terminal
            
            print("\nPipeline concluído. Gráficos 2D gerados e salvos como 'relatorio_complexidade_kmeans.png'.")

            # ==========================================
            # 6.5. Plotagem da Superfície 3D (Tempo vs n, k)
            # ==========================================
            fig_3d = plt.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            
            # Cria uma malha contínua para a superfície teórica baseada nos limites reais
            n_linspace = np.linspace(resumo['n'].min(), resumo['n'].max(), 30)
            k_linspace = np.linspace(resumo['k'].min(), resumo['k'].max(), 30)
            N_mesh, K_mesh = np.meshgrid(n_linspace, k_linspace)
            
            # Aplica a equação ajustada (popt) em todos os pontos da malha
            T_mesh = model_n_k((N_mesh, K_mesh), *popt)
            
            # Desenha a superfície teórica (Translúcida)
            surf = ax_3d.plot_surface(N_mesh, K_mesh, T_mesh, cmap='viridis', alpha=0.6, 
                                      linewidth=0.1, antialiased=True)
            fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, pad=0.1, label='Tempo (s)')
            
            # Sobrepõe os pontos empíricos medidos na bancada
            ax_3d.scatter(resumo['n'], resumo['k'], resumo['tempo_medio'], 
                          color='red', edgecolor='black', s=60, label='Pontos Empíricos', depthshade=False)
            
            # Configuração dos eixos e ângulos
            ax_3d.set_xlabel('Tamanho da Imagem (n pixels)')
            ax_3d.set_ylabel('Número de Clusters (K)')
            ax_3d.set_zlabel('Tempo de Execução (s)')
            ax_3d.set_title('Superfície F.O.C.A: Tempo(n, K)')
            ax_3d.view_init(elev=25, azim=135) # Ajuste o ângulo de visão conforme necessário
            ax_3d.legend()
            
            plt.tight_layout()
            plt.savefig('superficie_3d_foca.png', dpi=300)
            # plt.show() # Mantido comentado para não travar o terminal
            
            print("Gráfico 3D gerado e salvo como 'superficie_3d_foca.png'.")

        except Exception as e:
            print(f"\n[ERRO] Não foi possível ajustar a curva ou gerar os gráficos: {e}")
        
        # ==========================================
        # 7. Exportação dos Dados para Avaliação
        # ==========================================
        arquivo_csv = 'resultados_kmeans_aluno.csv'
        resumo.to_csv(arquivo_csv, index=False, float_format='%.6f', sep=';', decimal=',')
        print(f"Tabela de avaliação exportada com sucesso para: {arquivo_csv}")
    else:
        print("\nNenhum dado processado. Verifique se o diretório contém imagens válidas.")
        
    toc = time.time()
    print(f"\nTempo em: { toc - tic:.2f} segundos")