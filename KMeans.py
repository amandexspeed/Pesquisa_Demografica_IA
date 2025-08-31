from Preprocessamento import preprocessar_dados
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def aplicar_1_de_n(df, coluna):
    """Aplica a codificação One-Hot (1-de-n) a uma coluna categórica do DataFrame."""
    dummies = pd.get_dummies(df[coluna], prefix=coluna)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(coluna, axis=1)
    return df

def processar_dataFrame(df):
    """Processa o DataFrame para preparação para o K-Means."""
    df_processado = aplicar_1_de_n(df, 'continent')
    df_processado = aplicar_1_de_n(df_processado, 'region_y')
    return df_processado

def aplicar_kmeans(df, n_clusters):
    """Aplica o algoritmo K-Means ao DataFrame fornecido."""
    #Fazendo 1-de-n
    #df["continent"] = to_categorical(df["continent"].astype('category').cat.codes)

    df_processado = processar_dataFrame(df)

    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df_processado.select_dtypes(include=[np.number]))

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
    df_processado = processar_dataFrame(df)
    score = metrics.silhouette_score(df_processado.select_dtypes(include=[np.number]), df['Cluster'])
    
    print(f'Silhouette Score: {score}')
    
    print("\n--- Estatísticas Descritivas por Cluster ---")
    print(df.groupby('Cluster').describe())

    print("\n--- Contagem de Países por Cluster ---")
    print(df['Cluster'].value_counts())

    print("\n--- Anos analizados ---")
    print(pd.crosstab(df['Cluster'], df['year']))

    print("\n--- Análise de Continentes por Cluster ---")
    print(pd.crosstab(df['Cluster'], df['continent']))
    
    print("\n--- Análise de Regiões por Cluster ---")
    print(pd.crosstab(df['Cluster'], df['region_y']))

    print("\n--- Análise do PIB por Cluster ---")
    print(df.groupby('Cluster')['gdp_per_capita'].mean())

    print("\n--- Análise da Expectativa de Vida por Cluster ---")
    print(df.groupby('Cluster')['life_expectancy'].mean())

    print("\n--- Análise da População por Cluster ---")
    print(df.groupby('Cluster')['population'].mean())

    print("\n--- Análise do IDH por Cluster ---")
    print(df.groupby('Cluster')['hdi'].mean())

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
     

if __name__ == "__main__":
    main()