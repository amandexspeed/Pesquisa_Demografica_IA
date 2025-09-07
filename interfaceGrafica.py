from sklearn import cluster, metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Preprocessamento import preprocessar_dados
from KMeans import preparar_dataFrame, encontrar_k_otimo, aplicar_kmeans

def interpretar_resultados(df, df_original, n_clusters ,kmeans):

    cluster_info = df.groupby('Cluster')['hdi'].mean().sort_values(ascending=False)
    
    # Cria os nomes baseados no ranking de IDH
    nomes = ["Grupo A (IDH Muito Alto)", "Grupo B (IDH Alto)", "Grupo C (IDH Médio)", "Grupo D (IDH Baixo)", "Grupo E (IDH Muito Baixo)"]
    mapa_nomes = {cluster_id: nome for cluster_id, nome in zip(cluster_info.index, nomes[:n_clusters])}
    
    df['Grupo'] = df['Cluster'].map(mapa_nomes)
    
    print("\n--- Nomes dos Clusters (baseado no IDH médio decrescente) ---")
    print(df.groupby('Grupo')['hdi'].mean().sort_values(ascending=False))

    df_numerico = preparar_dataFrame(df_original)
    df_numerico = df_numerico.select_dtypes(include=[np.number])
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
        hue=df['Grupo'], 
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
    df_pca = pd.DataFrame(dados_pca, columns=['PC1', 'PC2'], index=df.index)
    df_pca['Grupo'] = df['Grupo']
    df_pca['country'] = df['region_y']  # Usando 'region_y' como proxy para o nome do país
    
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

def main():
    df_dataset = preprocessar_dados();
    if df_dataset is None:
        print("Erro no pré-processamento dos dados. Encerrando.")
        return
    df_processado = preparar_dataFrame(df_dataset)
    n_clusters = encontrar_k_otimo(df_processado.select_dtypes(include=[np.number]))  # Defina o número de clusters desejado
    df_clustered, kmeans = aplicar_kmeans(df_dataset, n_clusters)
    interpretar_resultados(df_clustered, df_dataset, n_clusters,kmeans)


if __name__ == "__main__":
    main()