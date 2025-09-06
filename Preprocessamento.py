import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns


remover_outliers = False
url_DadosGlobais = "https://raw.githubusercontent.com/amandexspeed/Pesquisa_Demografica_IA/refs/heads/main/DadosGlobais.csv"
url_PIB = "https://raw.githubusercontent.com/amandexspeed/Pesquisa_Demografica_IA/refs/heads/main/PIB.csv"

def carregar_dados(url_PIB=url_PIB, url_DadosGlobais=url_DadosGlobais):
    """Carrega os datasets de treinamento e teste a partir de URLs."""
    try:
        df_PIB = pd.read_csv(url_PIB, sep=';', encoding='utf-8')
        df_DadosGlobais = pd.read_csv(url_DadosGlobais, sep=';', encoding='utf-8')
        # CORREÇÃO: Emoji removido do print
        print(f"[SUCESSO] Base de treinamento carregada com sucesso!")
        return df_PIB, df_DadosGlobais
    except Exception as e:
        # CORREÇÃO: Emoji removido do print
        print(f"[ERRO] Erro ao carregar os dados: {e}")
        return None, None

def mesclar_dados(df_PIB, df_DadosGlobais):
    """Mescla os dois DataFrames com base na coluna 'country_code3'."""
    if df_PIB is None or df_DadosGlobais is None:
        return None
    df_merged = pd.merge(df_PIB, df_DadosGlobais, on='country_code3', how='inner')
    # CORREÇÃO: Emoji removido do print
    print(f"[SUCESSO] Dados mesclados com sucesso!")
    return df_merged

def converter_Decimal_para_numerico(df_merged, numeric_cols):
    """Converte colunas com vírgula decimal para o formato numérico."""
    # CORREÇÃO: Emoji removido do print
    print("[INFO] Convertendo colunas para formato numérico...")
    for col in numeric_cols:
        if col in df_merged.columns and df_merged[col].dtype == object:
            df_merged[col] = df_merged[col].astype(str).str.replace(',', '.', regex=False)
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

def preprocessar_dados():
    """Realiza o pré-processamento dos dados."""
    global remover_outliers
    print("--- INICIANDO PRÉ-PROCESSAMENTO ---")
    df_PIB, df_DadosGlobais = carregar_dados()
    if df_PIB is None or df_DadosGlobais is None:
        return None
    
    # CORREÇÃO: Emoji removido do print
    print("[INFO] Removendo valores nulos e duplicados...")
    df_PIB = df_PIB.dropna()
    df_DadosGlobais = df_DadosGlobais.dropna()
    df_PIB = df_PIB.drop_duplicates()
    df_DadosGlobais = df_DadosGlobais.drop_duplicates()

    df_dataset = mesclar_dados(df_PIB, df_DadosGlobais)
    if df_dataset is None: return None

    num_cols = ['gdp_per_capita', 'gdp_variation', 'population','gni','hdi', 'life_expectancy','expected_years_of_schooling', 'mean_years_of_schooling','gni']
    
    converter_Decimal_para_numerico(df_dataset, num_cols)
    df_dataset.dropna(subset=num_cols, inplace=True)

    # CORREÇÃO: Emoji removido do print
    print("[INFO] Filtrando dados a partir do ano de 2019...")
    df_dataset = df_dataset[df_dataset['year'] >= 2019].copy()

    colunas_numericas = df_dataset.select_dtypes(include=[np.number]).columns
    
    # CORREÇÃO: Emoji removido do print
    print("[INFO] Normalizando dados numéricos (MinMaxScaler)...")
    scaler = preprocessing.MinMaxScaler()
    df_dataset.loc[:, colunas_numericas] = scaler.fit_transform(df_dataset[colunas_numericas])

    if remover_outliers:
        print("[INFO] Removendo outliers com IsolationForest...")
        outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        outliers_dados = outlier_detector.fit_predict(df_dataset[colunas_numericas]) == -1
        df_dataset = df_dataset[~outliers_dados]

    # CORREÇÃO: Emoji removido do print
    print(f"[SUCESSO] Pré-processamento concluído! O dataset final tem {len(df_dataset)} registros.")
    return df_dataset

# --- NOVA FUNÇÃO PARA GERAR GRÁFICOS 
def gerar_visualizacoes(df):
    """
    Gera um painel com gráficos para analisar o DataFrame processado.
    """
    if df is None or df.empty:
        # CORREÇÃO: Emoji removido do print
        print("\n[ERRO] O DataFrame está vazio ou não foi gerado. Não é possível criar gráficos.")
        return

    # CORREÇÃO: Emoji removido do print
    print("\n[INFO] Gerando visualizações gráficas...")
    
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análise Gráfica dos Dados Socioeconômicos (Pós-Processamento)', fontsize=20)

    # 1. Histograma do PIB per capita
    sns.histplot(data=df, x='gdp_per_capita', kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribuição do PIB per capita Normalizado', fontsize=14)
    axes[0, 0].set_xlabel('PIB per capita (normalizado)')
    axes[0, 0].set_ylabel('Frequência')

    # 2. Histograma da Expectativa de Vida
    sns.histplot(data=df, x='life_expectancy', kde=True, ax=axes[0, 1], color='salmon')
    axes[0, 1].set_title('Distribuição da Expectativa de Vida Normalizada', fontsize=14)
    axes[0, 1].set_xlabel('Expectativa de Vida (normalizada)')
    axes[0, 1].set_ylabel('Frequência')

    # 3. Gráfico de Dispersão: PIB vs Expectativa de Vida
    sns.scatterplot(data=df, x='gdp_per_capita', y='life_expectancy', ax=axes[1, 0], alpha=0.6)
    sns.regplot(data=df, x='gdp_per_capita', y='life_expectancy', ax=axes[1, 0], scatter=False, color='red')
    axes[1, 0].set_title('Relação entre PIB e Expectativa de Vida', fontsize=14)
    axes[1, 0].set_xlabel('PIB per capita (normalizado)')
    axes[1, 0].set_ylabel('Expectativa de Vida (normalizada)')

    # 4. Boxplot de Indicadores Chave
    cols_boxplot = ['gdp_per_capita', 'life_expectancy', 'hdi', 'mean_years_of_schooling']
    sns.boxplot(data=df[cols_boxplot], ax=axes[1, 1])
    axes[1, 1].set_title('Boxplot de Indicadores-Chave', fontsize=14)
    axes[1, 1].set_ylabel('Valor Normalizado')
    plt.xticks(rotation=15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()


def main():
    """
    Função principal que executa o pré-processamento e a visualização.
    """
    df_dataset = preprocessar_dados()
    
    gerar_visualizacoes(df_dataset)
    
if __name__ == "__main__":
    main()