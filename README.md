🏦 Previsão de Churn Bancário com Machine Learning — Projeto End-to-End com XGBoost
1. Descrição

Este projeto tem como objetivo prever a evasão de clientes bancários (Churn Prediction) utilizando técnicas de Machine Learning. A solução foi construída de ponta a ponta, passando por todas as etapas do pipeline de Ciência de Dados: coleta, tratamento, análise exploratória, engenharia de atributos, modelagem preditiva e geração de insights estratégicos.

O modelo desenvolvido é capaz de identificar clientes com maior probabilidade de deixar o banco, permitindo ações preventivas para retenção e redução de perdas financeiras.

A solução foi construída utilizando o algoritmo XGBoost Classifier, um dos modelos mais poderosos para problemas de classificação em dados tabulares.

2. Problema de Negócio
2.1 Contexto da Empresa

A evasão de clientes é um dos maiores desafios enfrentados pelo setor bancário. Quando um cliente encerra seu relacionamento com o banco, ocorre impacto direto em:

Receita
Rentabilidade
Crescimento da carteira
Lifetime Value (LTV)
Estratégias comerciais

Além disso, conquistar novos clientes costuma ser significativamente mais caro do que manter clientes atuais.

2.2 Questão de Negócio

O banco necessita de uma solução capaz de prever quais clientes possuem maior risco de churn antes que a evasão aconteça.

A principal pergunta de negócio é:

Quais clientes possuem maior probabilidade de deixar o banco?

A resposta para essa pergunta permite:

Criar campanhas de retenção
Direcionar ações comerciais
Reduzir perdas financeiras
Melhorar relacionamento com clientes
Apoiar tomadas de decisão orientadas por dados
3. Entendimento do Negócio

O problema foi tratado como um caso de Classificação Supervisionada, onde:

A variável alvo (Exited) representa se o cliente deixou o banco.
O modelo aprende padrões comportamentais e financeiros para prever novos casos de evasão.

A proposta da solução é fornecer inteligência analítica para áreas como:

CRM
Retenção
Relacionamento
Marketing
Gestão estratégica
4. Coleta dos Dados

A base de dados foi obtida através da plataforma Kaggle.

Arquivo utilizado
Customer-Churn-Records.csv

O dataset contém informações comportamentais, financeiras e cadastrais dos clientes bancários.

5. Principais Variáveis do Dataset
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
6. Limpeza e Preparação dos Dados

A preparação dos dados foi realizada utilizando Python e Pandas.

Principais etapas realizadas:
6.1 Remoção de Colunas Irrelevantes

As colunas abaixo foram removidas por possuírem apenas função de identificação:

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)

Essas variáveis não agregavam valor preditivo ao modelo.

6.2 Tratamento de Variáveis Categóricas

As variáveis categóricas foram convertidas em variáveis numéricas utilizando One Hot Encoding.

df = pd.get_dummies(df, drop_first=True)

Essa técnica permite que algoritmos de Machine Learning interpretem corretamente dados textuais.

6.3 Separação entre Features e Target
X = df.drop('Exited', axis=1)
y = df['Exited']

Onde:

X → Variáveis explicativas
y → Variável alvo (churn)
7. Análise Exploratória dos Dados (EDA)

A análise exploratória foi utilizada para identificar padrões de comportamento relacionados à evasão.

Principais análises realizadas:
Distribuição de churn
Correlação entre variáveis
Relação entre atividade do cliente e evasão
Impacto do saldo bancário
Influência da idade e tempo de relacionamento
Comparação entre clientes ativos e inativos
8. Feature Engineering

Uma das etapas mais importantes do projeto foi a criação de novas variáveis derivadas para aumentar o poder preditivo do modelo.

Feature criada:
X['Balance_Activity_Ratio'] = X['Balance'] / (X['IsActiveMember'] + 1)
Objetivo da Feature

Essa variável busca identificar:

Relação entre saldo bancário e atividade do cliente
Nível de engajamento financeiro
Possíveis sinais de desinteresse no banco

A engenharia de atributos permitiu aumentar a capacidade de generalização do modelo.

9. Modelagem dos Dados e Machine Learning
9.1 Separação entre Treino e Teste

A base foi dividida em treino e teste utilizando train_test_split.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
Explicação dos parâmetros:
Parâmetro	Função
test_size=0.2	20% dos dados para teste
random_state=42	Garantir reprodutibilidade
9.2 Algoritmo Escolhido

O modelo utilizado foi o:

🌳 XGBoost Classifier
modelo_xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
9.3 Por que utilizar XGBoost?

O XGBoost é um dos algoritmos mais utilizados em problemas de classificação tabular devido à sua alta capacidade preditiva.

Principais vantagens:
✅ Alta performance
✅ Excelente generalização
✅ Redução de overfitting
✅ Treinamento eficiente
✅ Robustez para dados tabulares
✅ Forte utilização no mercado de Data Science
9.4 Como o XGBoost Funciona

O algoritmo utiliza a técnica de:

🔹 Gradient Boosting

O modelo cria várias árvores de decisão sequenciais.

Cada nova árvore tenta corrigir os erros cometidos pela árvore anterior.

Árvore 1 → Primeiras previsões
Árvore 2 → Corrige erros
Árvore 3 → Refina previsões
...
Modelo Final → Resultado consolidado

Essa abordagem torna o modelo altamente eficiente para previsão de churn.

10. Avaliação do Modelo

O desempenho do modelo foi avaliado utilizando métricas clássicas de classificação.

10.1 Accuracy
accuracy_score(y_test, y_pred)

Mede o percentual total de acertos do modelo.

10.2 Classification Report
classification_report(y_test, y_pred)

Inclui métricas importantes como:

Precision
Recall
F1-Score
10.3 Matriz de Confusão
confusion_matrix(y_test, y_pred)

A matriz de confusão permite identificar:

Resultado	Significado
Verdadeiro Positivo	Churn previsto corretamente
Verdadeiro Negativo	Cliente retido corretamente
Falso Positivo	Modelo previu churn incorretamente
Falso Negativo	Cliente saiu sem previsão
11. Importância das Variáveis

O XGBoost permite analisar quais variáveis tiveram maior impacto nas previsões.

modelo_xgb.feature_importances_
Benefícios dessa análise
Interpretabilidade do modelo
Entendimento do comportamento dos clientes
Identificação dos principais fatores de evasão
Apoio estratégico para retenção
12. Insights Estratégicos

O projeto permitiu identificar padrões relevantes no comportamento dos clientes bancários.

Principais insights encontrados:
Clientes menos ativos possuem maior probabilidade de churn
Saldo bancário influencia diretamente na retenção
Clientes com menor tempo de relacionamento apresentam maior risco de evasão
Clientes mais engajados possuem menor probabilidade de sair
Variáveis financeiras e comportamentais tiveram alto poder preditivo
13. Salvamento do Modelo

O modelo treinado foi serializado utilizando Joblib.

joblib.dump(modelo_xgb, 'modelo_xgboost.pkl')

Isso permite futura integração em:

APIs
Sistemas bancários
Dashboards
Streamlit
Aplicações Web
Serviços de predição em tempo real
14. Resultados do Projeto

O projeto demonstra como técnicas de Machine Learning podem gerar valor estratégico no setor financeiro.

Benefícios para o negócio:
✅ Antecipação de churn
✅ Redução de perdas financeiras
✅ Apoio à tomada de decisão
✅ Inteligência sobre comportamento dos clientes
✅ Estratégias de retenção mais eficientes
✅ Aplicação prática de Ciência de Dados
15. Próximos Passos e Melhorias

O projeto pode evoluir com novas melhorias:

Adicionar novas features comportamentais
Testar algoritmos como LightGBM e CatBoost
Implementar validação cruzada
Criar dashboard interativo
Desenvolver API REST
Deploy em nuvem (Render, AWS, Azure)
Implementar monitoramento do modelo
Automatizar pipeline de treinamento
16. Ferramentas Utilizadas
16.1 Linguagem e Ambiente
Linguagem: Python
Ambiente: Jupyter Notebook
16.2 Principais Bibliotecas
Manipulação de Dados
pandas
numpy
Visualização
matplotlib
seaborn
Machine Learning
scikit-learn
xgboost
joblib
17. Estrutura do Projeto
📦 churn-bank-prediction
│
├── 📄 Customer-Churn-Records.csv
├── 📄 resultado_churn.csv
├── 📄 modelo_xgboost.pkl
├── 📄 Untitled.ipynb
├── 📄 requirements.txt
└── 📄 README.md
18. Sobre o Projeto

Este projeto foi desenvolvido como parte de um portfólio de Ciência de Dados, demonstrando a construção de um pipeline completo de Machine Learning aplicado ao setor financeiro.

O trabalho envolve desde o tratamento e análise dos dados até a construção de um modelo preditivo robusto para previsão de churn bancário.

⭐ Diferenciais do Projeto
✅ Pipeline completo de Ciência de Dados
✅ Modelagem com XGBoost
✅ Engenharia de Features
✅ Insights estratégicos de negócio
✅ Estrutura organizada
✅ Código reproduzível
✅ Aplicação prática em cenário real
✅ Projeto preparado para futuras integrações
👨‍💻 Autor

Tulio Silva dos Santos

Profissional orientado a dados com foco em:

Ciência de Dados
Machine Learning
Engenharia de Dados
Automação de Processos
Business Intelligence
📬 Contato
LinkedIn: LinkedIn - Tulio Santos
GitHub: GitHub - tuliosannttos
✅ Status do Projeto

Projeto concluído e funcional.
Pronto para evoluções futuras, deploy em produção e integração com aplicações web e dashboards analíticos.
