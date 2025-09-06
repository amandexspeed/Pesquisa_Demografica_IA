from Preprocessamento import preprocessar_dados

from sklearn import cluster, metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

## 1. PREPARAÇÃO E PROCESSAMENTO DOS DADOS
# ----------------------------------------------------

def preparar_para_cluster(df):
    """
    Prepara o DataFrame para o K-Means, aplicando One-Hot Encoding 
    e retornando um DataFrame puramente numérico.
    """
    df_processado = df.copy()
    for coluna in ['continent', 'region_y']:
        dummies = pd.get_dummies(df_processado[coluna], prefix=coluna, drop_first=True)
        df_processado = pd.concat([df_processado.drop(coluna, axis=1), dummies], axis=1)
    return df_processado.select_dtypes(include=[np.number])


## 2. ENCONTRANDO O NÚMERO ÓTIMO DE CLUSTERS (K)
# ----------------------------------------------------

def encontrar_k_otimo(df_numerico, max_k=10):
    """
    Usa o Método do Cotovelo para encontrar o número ideal de clusters (k).
    """
    print("[INFO] Calculando o número ótimo de clusters com o Método do Cotovelo...")
    inercias = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = cluster.KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_numerico)
        inercias.append(kmeans.inertia_)
        
    # --- GRÁFICA DO COTOVELO ---
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=k_range, y=inercias, marker='o', color='royalblue', linewidth=2, markersize=8)
    plt.title('Método do Cotovelo para Escolha do K Ideal', fontsize=16)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inércia (Soma das distâncias quadráticas)', fontsize=12)
    plt.xticks(k_range)
    # Adiciona uma anotação para guiar o usuário
    plt.annotate(
        'Ponto de inflexão (Cotovelo)', 
        xy=(3, inercias[1]),
        xytext=(4, inercias[1] + 5), 
        arrowprops=dict(facecolor='black', shrink=0.05),
        fontsize=12
    )
    plt.show()
    
    print("\n[INSTRUÇÃO] Observe o gráfico acima. O 'cotovelo' (ponto onde a curva se torna menos íngreme)")
    print("indica o número ótimo de clusters. Insira o valor que você considera ideal.")
    k_otimo = int(input("Digite o número de clusters (k) escolhido: "))
    return k_otimo


## 3. APLICAÇÃO DO K-MEANS E VISUALIZAÇÃO INTUITIVA 
# ----------------------------------------------------

def aplicar_kmeans_e_visualizar(df_original, df_numerico, n_clusters):
    """
    Aplica K-Means, interpreta os clusters e gera uma visualização rica e intuitiva.
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df_numerico)
    df_original['Cluster'] = cluster_labels
    
    # --- INTERPRETAÇÃO E NOMEAÇÃO AUTOMÁTICA DOS CLUSTERS ---
    # Calcula o IDH médio para cada cluster para criar nomes significativos
    cluster_info = df_original.groupby('Cluster')['hdi'].mean().sort_values(ascending=False)
    
    # Cria os nomes baseados no ranking de IDH
    nomes = ["Grupo A (IDH Muito Alto)", "Grupo B (IDH Alto)", "Grupo C (IDH Médio)", "Grupo D (IDH Baixo)", "Grupo E (IDH Muito Baixo)"]
    mapa_nomes = {cluster_id: nome for cluster_id, nome in zip(cluster_info.index, nomes[:n_clusters])}
    
    df_original['Grupo'] = df_original['Cluster'].map(mapa_nomes)
    
    print("\n--- Nomes dos Clusters (baseado no IDH médio decrescente) ---")
    print(df_original.groupby('Grupo')['hdi'].mean().sort_values(ascending=False))
    
    # --- VISUALIZAÇÃO MELHORADA COM PCA ---
    print(f"\n[INFO] Gerando visualização para {n_clusters} clusters usando PCA...")
    pca = PCA(n_components=2)
    dados_pca = pca.fit_transform(df_numerico)
    centroides_pca = pca.transform(kmeans.cluster_centers_)
    
    plt.figure(figsize=(15, 10))
    # Usando uma paleta de cores mais viva e distinta
    palette = sns.color_palette("bright", n_clusters)
    
    # Gráfico de dispersão
    sns.scatterplot(
        x=dados_pca[:, 0], 
        y=dados_pca[:, 1], 
        hue=df_original['Grupo'], 
        hue_order=nomes[:n_clusters], 
        palette=palette,
        s=80,
        alpha=0.9
    )
    
    # Centroides
    plt.scatter(
        x=centroides_pca[:, 0], 
        y=centroides_pca[:, 1], 
        s=400, 
        c='black', 
        marker='*',
        label='Centroides',
        edgecolor='white'
    )
    
    # --- ANOTAÇÃO DE PAÍSES NO GRÁFICO ---
    # Adiciona o nome de alguns países para dar contexto
    df_pca = pd.DataFrame(dados_pca, columns=['PC1', 'PC2'], index=df_original.index)
    df_pca['Grupo'] = df_original['Grupo']
    df_pca['country'] = df_original['country']
    
    for grupo in df_pca['Grupo'].unique():
        # Pega 2 países de exemplo para cada grupo
        amostra = df_pca[df_pca['Grupo'] == grupo].sample(min(2, len(df_pca[df_pca['Grupo'] == grupo])), random_state=42)
        for i, row in amostra.iterrows():
            plt.text(row['PC1'] + 0.01, row['PC2'], row['country'], fontsize=9, color='black', style='italic')

    # Títulos e legendas mais descritivos
    plt.title(f'Agrupamento de Países por Similaridade Socioeconômica ({n_clusters} Grupos)', fontsize=18, weight='bold')
    plt.xlabel('Componente Principal 1 (Eixo de Desenvolvimento Econômico/Social)', fontsize=12)
    plt.ylabel('Componente Principal 2 (Eixo de Variação Secundária)', fontsize=12)
    plt.legend(title='Grupos Socioeconômicos', fontsize=10, title_fontsize=12)
    plt.grid(linestyle='--', alpha=0.6)
    plt.show()
    
    return df_original, df_numerico


## 4. AVALIAÇÃO DO MODELO
# ----------------------------------------------------

def avaliar_modelo(df_com_clusters, df_numerico):
    """
    Calcula o Silhouette Score e imprime análises detalhadas dos clusters.
    """
    score = metrics.silhouette_score(df_numerico, df_com_clusters['Cluster'])
    print("\n--- AVALIAÇÃO DO MODELO DE CLUSTERIZAÇÃO ---")
    print(f'Silhouette Score: {score:.4f} (Quanto mais próximo de 1, melhor)')
    
    print("\n--- Contagem de Países por Grupo ---")
    print(df_com_clusters['Grupo'].value_counts().sort_index())

    print("\n--- Análise de Continentes por Grupo ---")
    print(pd.crosstab(df_com_clusters['Grupo'], df_com_clusters['continent']))
    
    print("\n--- Média de Indicadores por Grupo ---")
    colunas_analise = ['gdp_per_capita', 'life_expectancy', 'hdi', 'population']
    print(df_com_clusters.groupby('Grupo')[colunas_analise].mean().round(2))

## 5. FLUXO PRINCIPAL
# ----------------------------------------------------

def main():
    """
    Orquestra o fluxo completo: pré-processamento, otimização de k,
    clusterização, visualização e avaliação.
    """
    df_dataset = preprocessar_dados()
    if df_dataset is None or df_dataset.empty:
        print("[ERRO] Pré-processamento falhou ou resultou em um DataFrame vazio.")
        return
    
    df_numerico = preparar_para_cluster(df_dataset)
    
    n_clusters_otimo = encontrar_k_otimo(df_numerico)
    
    df_final, df_numerico_final = aplicar_kmeans_e_visualizar(df_dataset, df_numerico, n_clusters_otimo)
    
    avaliar_modelo(df_final, df_numerico_final)

if __name__ == "__main__":
    main()