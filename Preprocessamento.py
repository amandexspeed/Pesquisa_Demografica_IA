import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import IsolationForest

remover_outliers = False
url_DadosGlobais = "https://raw.githubusercontent.com/amandexspeed/Pesquisa_Demografica_IA/refs/heads/main/DadosGlobais.csv"
url_PIB = "https://raw.githubusercontent.com/amandexspeed/Pesquisa_Demografica_IA/refs/heads/main/PIB.csv"

def carregar_dados(url_PIB=url_PIB, url_DadosGlobais=url_DadosGlobais):
    """Carrega os datasets de treinamento e teste a partir de URLs."""
    try:
        # Carrega os dados usando pandas com separador vírgula e codificação UTF-8
        # Isso garante que os dados sejam lidos corretamente, especialmente se contiverem caracteres especiais
        df_PIB = pd.read_csv(url_PIB, sep=',',encoding='utf-8')
        df_DadosGlobais = pd.read_csv(url_DadosGlobais, sep=',',encoding='utf-8')
        print(f"Base de treinamento carregada com sucesso !") 
        return df_PIB, df_DadosGlobais
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None, None
    
def mesclar_dados(df_PIB, df_DadosGlobais):
    """Mescla os dois DataFrames com base na coluna 'country_code3'."""
    if df_PIB is None or df_DadosGlobais is None:
        print("Erro: Dados não carregados corretamente.")
        return None
    
    
    df_merged = pd.merge(df_PIB, df_DadosGlobais, on='country_code3', how='inner')
    print(f"Dados mesclados com sucesso !")
    return df_merged

def converter_para_numerico(df_merged, numeric_cols):
    """Converte colunas específicas para o tipo numérico, lidando com erros."""
    for col in numeric_cols:
        if df_merged[col].dtype == object:
            df_merged[col] = df_merged[col].astype(str).str.replace('.', '', regex=False)
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')
            
    
    
def preprocessar_dados(df_PIB, df_DadosGlobais):
    global remover_outliers
    """Realiza o pré-processamento dos dados."""
    if df_PIB is None or df_DadosGlobais is None:
        print("Erro: Dados não carregados corretamente.")
        return None, None
    
    df_PIB = df_PIB.dropna()
    df_DadosGlobais = df_DadosGlobais.dropna()

    df_PIB = df_PIB.drop_duplicates()
    df_DadosGlobais = df_DadosGlobais.drop_duplicates()

    df_dataset = mesclar_dados(df_PIB, df_DadosGlobais)

    subset_cols = ['gdp_per_capita', 'gdp_variation', 'population', 'population_below_poverty_line',
                   'hdi', 'expected_years_of_schooling', 'mean_years_of_schooling',
                   'gni']
    
    #converter_para_numerico(df_dataset, subset_cols)

    # Seleciona apenas as colunas numéricas para o PCA
    colunas_numericas = df_dataset.select_dtypes(include=[np.number]).columns

    df_dataset = df_dataset[(df_dataset[['year']] >= 2019).all(axis=1)]
    print(df_dataset.info())

    # Normalização dos dados numéricos
    scaler = preprocessing.MinMaxScaler()
    df_dataset[colunas_numericas] = scaler.fit_transform(df_dataset[colunas_numericas])

    if remover_outliers:
        outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        outliers_dados = outlier_detector.fit_predict(df_dataset[colunas_numericas]) == -1

        df_dataset = df_dataset[~outliers_dados]

    print(f"Pré-processamento concluído com sucesso !")

    return df_dataset

def main():
    df_PIB, df_DadosGlobais = carregar_dados()
    df_dataset = preprocessar_dados(df_PIB, df_DadosGlobais)
    print(df_dataset.info())
    print(df_dataset["gdp_per_capita"])
    print(df_dataset["life_expectancy"])
    
if __name__ == "__main__":
    main()# -*- coding: utf-8 -*-