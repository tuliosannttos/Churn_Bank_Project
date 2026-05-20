# 🏦 Previsão de Churn Bancário com Machine Learning

📌 Problema de Negócio

A evasão de clientes (Churn) é um dos principais desafios do setor bancário, impactando diretamente receita, crescimento e retenção de clientes.

O objetivo deste projeto é desenvolver um modelo de Machine Learning capaz de prever quais clientes possuem maior probabilidade de deixar o banco, permitindo que a empresa atue de forma preventiva através de estratégias de retenção e redução de perdas financeiras.

🎯 Objetivos do Projeto

Este projeto foi desenvolvido para:

Prever a probabilidade de churn de clientes bancários
Identificar padrões comportamentais relacionados à evasão
Gerar insights estratégicos através dos dados
Aplicar técnicas de Machine Learning em um cenário real
Apoiar tomadas de decisão orientadas por dados
🧠 Estratégia da Solução

O projeto foi estruturado seguindo um pipeline completo de Ciência de Dados.

🔹 Etapa 01 — Coleta dos Dados

Importação da base de clientes bancários no site Kaggle.

🔹 Etapa 02 — Limpeza e Tratamento

Preparação dos dados para modelagem preditiva.

🔹 Etapa 03 — Análise Exploratória dos Dados (EDA)

Análise de padrões, correlações e comportamento dos clientes.

🔹 Etapa 04 — Feature Engineering

Criação de novas variáveis para aumentar o poder preditivo.

🔹 Etapa 05 — Modelagem de Machine Learning

Treinamento do modelo XGBoost Classifier.

🔹 Etapa 06 — Avaliação do Modelo

Análise das métricas de desempenho e capacidade preditiva.

🔹 Etapa 07 — Geração de Insights

Extração de insights estratégicos para o negócio.

📂 Estrutura do Projeto
📦 churn-bank-prediction
│
├── 📄 Customer-Churn-Records.csv
├── 📄 resultado_churn.csv
├── 📄 modelo_xgboost.pkl
├── 📄 Untitled.ipynb
├── 📄 requirements.txt
└── 📄 README.md
🛠️ Tecnologias Utilizadas
🔹 Linguagem
Python
🔹 Bibliotecas Principais
Pandas
NumPy
Matplotlib
Seaborn
Scikit-Learn
XGBoost
Joblib
📊 Sobre o Dataset

A base de dados contém informações de clientes bancários e seus comportamentos financeiros.

Principais Variáveis
Variável	Descrição
CreditScore	Score de crédito
Geography	País do cliente
Gender	Gênero
Age	Idade
Tenure	Tempo de relacionamento
Balance	Saldo bancário
NumOfProducts	Quantidade de produtos
HasCrCard	Possui cartão de crédito
IsActiveMember	Cliente ativo
EstimatedSalary	Salário estimado
Exited	Variável alvo (Churn)
🧹 Limpeza e Pré-Processamento dos Dados

Antes do treinamento do modelo, os dados passaram por etapas de preparação.

🔹 Remoção de Colunas Irrelevantes
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

Essas colunas foram removidas porque funcionam apenas como identificadores e não agregam valor preditivo ao modelo.

🔹 Tratamento de Variáveis Categóricas

As variáveis categóricas foram convertidas em variáveis numéricas utilizando One Hot Encoding.

df = pd.get_dummies(df, drop_first=True)

Isso permite que o algoritmo compreenda corretamente dados textuais.

🏗️ Feature Engineering

Uma das etapas mais importantes do projeto foi a criação de novas variáveis derivadas.

Exemplo:
X['Balance_Activity_Ratio'] = X['Balance'] / (X['IsActiveMember'] + 1)
🎯 Objetivo da Feature

Essa variável busca identificar:

Relação entre saldo bancário e atividade do cliente
Nível de engajamento financeiro
Possíveis sinais de desinteresse no banco

A engenharia de atributos aumenta significativamente a capacidade preditiva do modelo.

✂️ Separação entre Treino e Teste

A base foi dividida em conjuntos de treino e teste.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
Explicação dos Parâmetros
Parâmetro	Função
test_size=0.2	20% dos dados destinados ao teste
random_state=42	Reprodutibilidade dos resultados
🤖 Modelo de Machine Learning

O algoritmo escolhido foi:

🌳 XGBoost Classifier
modelo_xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
🧠 Por que utilizar XGBoost?

O XGBoost é um dos algoritmos mais poderosos para dados tabulares e problemas de classificação.

Principais Vantagens

✅ Alta performance preditiva
✅ Excelente capacidade de generalização
✅ Redução de overfitting
✅ Treinamento rápido
✅ Identificação de relações complexas
✅ Amplamente utilizado no mercado e competições de Data Science

⚙️ Como o XGBoost Funciona

O algoritmo utiliza a técnica:

🔹 Gradient Boosting

O modelo cria várias árvores de decisão em sequência.

Cada nova árvore tenta corrigir os erros da árvore anterior.

Árvore 1 → Primeiras previsões
Árvore 2 → Corrige erros
Árvore 3 → Refina resultados
...
Modelo Final → Previsão consolidada

Isso gera um sistema altamente preciso para classificação de churn.

📈 Avaliação do Modelo

O desempenho do modelo foi avaliado utilizando métricas de classificação.

🔹 Accuracy
accuracy_score(y_test, y_pred)

Mede o percentual de acertos do modelo.

🔹 Classification Report
classification_report(y_test, y_pred)

Inclui métricas importantes como:

Precision
Recall
F1-Score
🔹 Matriz de Confusão
confusion_matrix(y_test, y_pred)

A matriz permite analisar:

Resultado	Significado
Verdadeiro Positivo	Churn previsto corretamente
Verdadeiro Negativo	Cliente retido corretamente
Falso Positivo	Modelo previu churn incorretamente
Falso Negativo	Cliente saiu sem previsão do modelo
🔍 Importância das Variáveis

O XGBoost permite identificar quais variáveis mais influenciaram as previsões.

modelo_xgb.feature_importances_
Benefícios dessa análise
Interpretabilidade do modelo
Entendimento do comportamento dos clientes
Identificação dos principais fatores de evasão
Apoio estratégico para retenção
📈 Insights Estratégicos

O projeto permitiu identificar padrões relevantes no comportamento dos clientes.

Principais Insights
Clientes menos ativos possuem maior probabilidade de churn
Saldo bancário influencia diretamente na retenção
Clientes com menor tempo de relacionamento tendem a abandonar o banco com mais frequência
Clientes com maior engajamento possuem menor risco de evasão
Variáveis comportamentais e financeiras apresentaram alto poder preditivo
💾 Salvamento do Modelo

O modelo treinado foi salvo utilizando Joblib.

joblib.dump(modelo_xgb, 'modelo_xgboost.pkl')

Isso permite futura integração em:

APIs
Sistemas bancários
Streamlit
Dashboards
Aplicações Web
Serviços de predição em tempo real
🚀 Resultados do Projeto

O projeto demonstra como Machine Learning pode gerar valor estratégico no setor financeiro.

Benefícios para o Negócio

✅ Antecipação de churn
✅ Redução de perdas financeiras
✅ Apoio à tomada de decisão
✅ Inteligência sobre comportamento de clientes
✅ Estratégias de retenção mais eficientes
✅ Aplicação prática de Ciência de Dados

🌐 Melhorias Futuras

Possíveis evoluções do projeto:

Deploy com Streamlit
API com Flask/FastAPI
Otimização de hiperparâmetros
Pipeline automatizado
Monitoramento do modelo
Deploy em nuvem
Implementação de MLOps
▶️ Como Executar o Projeto
Clone o repositório
git clone https://github.com/seuusuario/churn-bank-prediction.git
Instale as dependências
pip install -r requirements.txt
Execute o Jupyter Notebook
jupyter notebook

Abra:

Untitled.ipynb
👨‍💻 Autor
Tulio Santos

Profissional orientado a dados com foco em:

Ciência de Dados
Machine Learning
Engenharia de Dados
Automação de Processos
Business Intelligence
📬 Contato

📧 tulio-320hotmail.com

🔗 https://www.linkedin.com/in/túlio-santos-b65720a4/

🐙 

⭐ Destaques do Projeto

Este projeto demonstra um pipeline completo de Machine Learning aplicado a um problema real do setor bancário, envolvendo:

Análise de Dados
Engenharia de Features
Modelagem Preditiva
Inteligência de Negócio
Machine Learning
Insights Estratégicos

Transformando dados em valor para o negócio através da previsão de churn de clientes.
