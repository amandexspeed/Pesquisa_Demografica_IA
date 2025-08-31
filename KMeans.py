from Preprocessamento import preprocessar_dados
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def aplicar_kmeans(df, n_clusters):
    """Aplica o algoritmo K-Means ao DataFrame fornecido."""
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df.select_dtypes(include=[np.number]))
    return df, kmeans

def plotar_clusters(df, kmeans):
    """Plota os clusters formados pelo K-Means."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=df.select_dtypes(include=[np.number]).columns[0], 
                    y=df.select_dtypes(include=[np.number]).columns[1], hue='Cluster', palette='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.show()

def avaliar_modelo(df):
    """Avalia o modelo K-Means usando o índice de silhueta."""
    score = metrics.silhouette_score(df.select_dtypes(include=[np.number]), df['Cluster'])
    print(f'Silhouette Score: {score}')
    return score

def main():
    df_dataset = preprocessar_dados()
    if df_dataset is None:
        print("Erro no pré-processamento dos dados. Encerrando.")
        return
    n_clusters = 5  # Defina o número de clusters desejado
    df_clustered, kmeans = aplicar_kmeans(df_dataset, n_clusters)
    plotar_clusters(df_clustered, kmeans)
    avaliar_modelo(df_clustered)
    print(df_clustered.info())

if __name__ == "__main__":
    main()