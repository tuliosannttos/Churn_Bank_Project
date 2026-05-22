import streamlit as st
import pandas as pd

# ==================================================
# CONFIGURAÇÃO DA PÁGINA
# ==================================================

st.set_page_config(

    page_title='Dashboard Churn Bancário',
    layout='wide'

)

# ==================================================
# TÍTULO
# ==================================================

st.title('Dashboard de Churn Bancário')

st.write(
    'Análise preditiva de clientes com risco de abandono.'
)

# ==================================================
# CARREGAR BASE FINAL
# ==================================================

df = pd.read_csv(
    'resultado_churn.csv'
)

# ==================================================
# VISUALIZAR DADOS
# ==================================================

st.subheader('Base de Dados')

st.dataframe(df.head())

# ==================================================
# MÉTRICAS PRINCIPAIS
# ==================================================

total_clientes = df.shape[0]

clientes_churn = df[
    df['Churn_Previsto'] == 1
].shape[0]

taxa_churn = (
    clientes_churn / total_clientes
) * 100

# ==================================================
# CARDS
# ==================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    'Total de Clientes',
    total_clientes
)

col2.metric(
    'Clientes com Risco de Churn',
    clientes_churn
)

col3.metric(
    'Taxa de Churn (%)',
    f'{taxa_churn:.2f}%'
)

# ==================================================
# TOP CLIENTES DE RISCO
# ==================================================

st.subheader(
    'Top Clientes com Maior Probabilidade de Churn'
)

top_risco = df.sort_values(
    by='Probabilidade_Churn',
    ascending=False
)

st.dataframe(

    top_risco[
        [
            'Age',
            'Balance',
            'CreditScore',
            'Probabilidade_Churn'
        ]
    ].head(20)

)

# ==================================================
# DISTRIBUIÇÃO DE CHURN
# ==================================================

st.subheader('Distribuição de Churn')

grafico = df['Churn_Previsto'].value_counts()

st.bar_chart(grafico)

# ==================================================
# DISTRIBUIÇÃO DAS PROBABILIDADES
# ==================================================

st.subheader(
    'Distribuição das Probabilidades de Churn'
)

st.line_chart(
    df['Probabilidade_Churn']
)

# ==================================================
# DOWNLOAD
# ==================================================

csv = df.to_csv(index=False)

st.download_button(

    label='Baixar Resultado CSV',

    data=csv,

    file_name='resultado_churn.csv',

    mime='text/csv'

)