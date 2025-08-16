# %% [markdown]
# # Preparação de Dados e Análise Econométrica
# 
# **Objetivo:** Ler dados de um Google Sheets, criar variáveis dummies para todas as variáveis de controle, 
# preservar consistência entre variáveis dependentes e independentes, e executar regressões com modelo logit ordenado.
# 

# %% [markdown]
# ## 1. Importando bibliotecas necessárias

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import seaborn as sns
import re
import textwrap

# %%
url = 'https://docs.google.com/spreadsheets/d/1kn6u7jnv-ktwiY0TMYxYWnQhr54dWOYSOYvrqf9W3WY/export?format=csv'
df = pd.read_csv(url)

# Visualização inicial
df.head()

# %% [markdown]
# ## 3. Tratamento das variáveis
# 
# **O que faremos:**  
# - Evitar codificação manual das variáveis de controle.  
# - Transformar todas as variáveis categóricas de controle em *dummies* automaticamente.  
# - Manter colunas numéricas como estão.

# %%
# Lista de variáveis de controle (categorias + numéricas)
variaveis_controle = [
    'Qual seu nível de formação em Ciências Econômicas? ',
    'Você é homem? ',
    'Com qual grupo racial/étnico você mais se identifica? ',
    'Qual seu nível de escolaridade? ',
    'Qual é a sua faixa etária? ',
    'Em qual região do Brasil você reside? ',
    'Qual é o seu vínculo empregatício? ',
    'Você trabalha com economia, finanças, contabilidade ou áreas correlatas? ',
    'Você trabalha com política ou áreas correlatas? ',
    'Você se considera uma pessoa politicamente engajada? ',
    'Você costuma acompanhar notícias sobre economia e política? ',
    'Com qual espectro político você mais se identifica? '
]

# Criando um DataFrame apenas com as variáveis de controle + as perguntas (dependentes)
df_controle = df[variaveis_controle].copy()

# Transformando TODAS as variáveis categóricas em dummies
# Por que: evita perda de informação ordinal/categórica e garante consistência no modelo
df_controle_dummies = pd.get_dummies(df_controle, drop_first=True)


# %% [markdown]
# ## 4. Tratamento das variáveis dependentes
# 
# Vamos identificar todas as perguntas (variáveis dependentes) que não são de controle e criar versão numérica.
# 

# %%
# Função auxiliar: detecta número na resposta (ex.: escala Likert)
def detect_number(row):
    if pd.isnull(row):
        return np.nan
    matches = re.findall(r'[0-9]', str(row))
    return min(map(int, matches)) if matches else np.nan

# Lista de colunas que NÃO entrarão como variáveis dependentes
colunas_excluidas = ['Carimbo de data/hora',
                     'Ao clicar em "Aceito", você declara que compreendeu as informações e consente em participar.'] + variaveis_controle

# Criando DataFrame com variáveis dependentes mapeadas
df_dependentes = df.drop(columns=colunas_excluidas).applymap(detect_number)

# Gerando acrônimos curtos para variáveis dependentes
def make_acronym(colname):
    col = re.sub(r'[^\w\s]', ' ', colname, flags=re.UNICODE)
    stopwords = {'de', 'da', 'do', 'das', 'dos', 'a', 'o', 'as', 'os', 'em', 'na', 'no', 
                 'nas', 'nos', 'para', 'por', 'com', 'ao', 'à', 'às', 'e', 'ou', 'que', 
                 'se', 'é', 'uma', 'você', 'qual', 'sua', 'são', 'sobre', 'tem'}
    tokens = [t.lower() for t in col.split() if t.lower() not in stopwords]
    return '_'.join(tokens[:3]) if len(tokens) >= 2 else (tokens[0] if tokens else col.lower())

rename_dict = {col: make_acronym(col) for col in df_dependentes.columns}
df_dependentes.rename(columns=rename_dict, inplace=True)

# %% [markdown]
# ## 5. Juntando variáveis de controle e dependentes

# %%
df_final = pd.concat([df_dependentes, df_controle_dummies], axis=1)

# %% [markdown]
# ## 6. Função de análise (Modelo Logit Ordenado)
# 
# **O que faz:**  
# - Filtra a variável dependente desejada + controles.  
# - Gera modelo logit ordenado.  
# - Calcula médias previstas e reais para diferentes públicos.  
# - Gera gráfico comparativo.
# 

# %%
def analisar_variavel_para_latex(df, variavel):
    print(f"\nIniciando análise da variável '{variavel}'")
    
    # Filtra colunas: variável dependente + todas as colunas de controle já dummies
    colunas_modelo = [variavel] + [c for c in df.columns if c != variavel]
    df_model = df[colunas_modelo].dropna()
    
    X = df_model.drop(columns=[variavel])
    y = df_model[variavel]
    
    # Criando modelo logit ordenado
    model = OrderedModel(y, X, distr='logit', hasconst=False)
    result = model.fit(method='lbfgs')
    
    print(result.summary())
    
    # Exemplo de cálculo: comparar público com dummy 'Você é homem? _Sim'
    if 'Você é homem? _Sim' in X.columns:
        grupo1 = df_model[df_model['Você é homem? _Sim'] == 1][variavel].mean()
        grupo0 = df_model[df_model['Você é homem? _Sim'] == 0][variavel].mean()
    else:
        grupo1 = grupo0 = np.nan
    
    # Gráfico comparativo
    plt.figure(figsize=(6,4))
    sns.barplot(x=[grupo0, grupo1], y=['Grupo 0', 'Grupo 1'], color='blue', alpha=0.3, edgecolor='blue')
    plt.title(f"Média por grupo - {variavel}")
    plt.xlim(0, 2)
    for i, v in enumerate([grupo0, grupo1]):
        plt.text(v + 0.05, i, f'{v:.2f}', va='center')
    plt.show()

# %% [markdown]
# ## 7. Executando análises para todas as variáveis dependentes

# %%
for variavel in df_dependentes.columns:
    analisar_variavel_para_latex(df_final, variavel)


