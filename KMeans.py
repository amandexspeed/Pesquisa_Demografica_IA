import easygui
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

def preparar_dataFrame(df):
    """Prepara o DataFrame para poder ser usado para o K-Means."""
    df_copia = df.copy()
    aplicar_1_de_n(df_copia, 'continent')
    aplicar_1_de_n(df_copia, 'region_y')
    return df_copia

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
    
    return obter_k_graficamente()

def obter_k_graficamente():
    msg = "Observe o gráfico do 'Método do Cotovelo'.\n\nDigite o número de clusters (k) que você considera ideal:"
    title = "Escolha do K Ideal"
    
    k_otimo = easygui.integerbox(msg, title, lowerbound=1) # lowerbound garante que seja > 0
    return k_otimo

def aplicar_kmeans(df, n_clusters):
    """Aplica o algoritmo K-Means ao DataFrame fornecido."""
    df_copia = df.copy()
    df_processado = preparar_dataFrame(df_copia)
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    df_copia['Cluster'] = kmeans.fit_predict(df_processado.select_dtypes(include=[np.number]))

    return df_copia, kmeans

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
    #df_processado = preparar_dataFrame(df)
    score = metrics.silhouette_score(df.select_dtypes(include=[np.number]), df['Cluster'])
    
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
    df_processado = preparar_dataFrame(df_dataset)
    n_clusters = encontrar_k_otimo(df_processado.select_dtypes(include=[np.number]))  # Defina o número de clusters desejado
    df_clustered, kmeans = aplicar_kmeans(df_dataset, n_clusters)
    plotar_clusters(df_clustered, kmeans)
    avaliar_modelo(df_clustered)
     

if __name__ == "__main__":
    main()