*Projeto de Previsão de Churn de Clientes - Banco*



##1. Descrição

Este é um projeto end-to-end de Machine Learning para prever a evasão de clientes (churn) em uma instituição financeira. O objetivo é identificar clientes com alta probabilidade de deixar o banco, permitindo a criação de ações de retenção estratégicas.

O projeto segue as etapas do CRISP-DM:

Entendimento do Negócio

Coleta e Limpeza dos Dados

Análise Exploratória (EDA)

Feature Engineering

Modelagem e Seleção do Algoritmo (XGBoost)

Avaliação e Interpretação do Modelo

Geração de Insights para Tomada de Decisão

***##2. Problema de Negócio***
2.1 Contexto
O setor bancário enfrenta uma alta competitividade, onde reter clientes existentes é tão crucial quanto adquirir novos. A perda de clientes (churn) impacta diretamente a receita e aumenta os custos de aquisição.

2.2 Questão de Negócio
A instituição financeira precisa reduzir sua taxa de evasão. Para isso, deseja identificar proativamente os clientes com maior risco de deixar o banco, antes que a saída se concretize.

O desafio é: como prever o churn com base nos dados demográficos, de transação e perfil dos clientes?

💡 3. Entendimento do Negócio
O objetivo do projeto é fornecer um modelo de Machine Learning que classifique clientes como potenciais churners ou não. Com essa visibilidade, o time de marketing e relacionamento pode:

Oferecer benefícios personalizados para aumentar o engajamento.

Realizar campanhas de retenção com foco nos clientes de alto risco.

Alocar recursos de forma mais eficiente, evitando gastos com clientes satisfeitos.

📊 4. Coleta dos Dados
Os dados utilizados neste projeto são de natureza simulada, mas representam um cenário real de uma instituição financeira. O dataset (Customer-Churn-Records.csv) contém 10.000 registros de clientes e as seguintes features:

Feature	Descrição
RowNumber	Índice da linha no arquivo
CustomerId	Identificador único do cliente
Surname	Sobrenome do cliente
CreditScore	Pontuação de crédito
Geography	Localização geográfica (França, Espanha, Alemanha)
Gender	Gênero (Masculino, Feminino)
Age	Idade do cliente
Tenure	Tempo como cliente do banco (anos)
Balance	Saldo atual na conta
NumOfProducts	Número de produtos contratados
HasCrCard	Possui cartão de crédito? (1 = Sim / 0 = Não)
IsActiveMember	Membro ativo? (1 = Sim / 0 = Não)
EstimatedSalary	Salário estimado
Exited	Target (Variável Alvo): Cliente saiu? (1 = Sim / 0 = Não)
🧹 5. Limpeza e Preparação dos Dados
Nesta etapa realizamos:

Verificação de Nulos: Confirmado que não há valores ausentes no dataset.

Remoção de Colunas: Colunas que poderiam causar data leakage ou que não agregam valor preditivo (Complain, Satisfaction Score, Point Earned) foram removidas.

Codificação: Variáveis categóricas (Geography, Gender, Card Type) foram transformadas em formato numérico utilizando LabelEncoder do scikit-learn.

📈 6. Análise Exploratória (EDA) & Feature Engineering
A EDA nos permitiu entender o comportamento dos clientes e criar novas variáveis para melhorar o poder de predição do modelo.

6.1 Feature Engineering
Foram criadas novas features com base nas relações de negócio:

Saldo_Atividade: Relação entre o saldo e a inatividade (Balance / (IsActiveMember + 1)). Clientes inativos com alto saldo podem ser um sinal de alerta.

Produtos_Idade: Número de produtos em relação à idade (NumOfProducts / Age).

Score_Idade: Pontuação de crédito em relação à idade (CreditScore / Age).

6.2 Insights Iniciais (Hipóteses)
Produtos: Clientes com apenas 1 produto são mais propensos a sair por não estarem engajados.

Idade: Clientes mais jovens (faixa dos 30-40 anos) podem ter maior mobilidade e propensão a trocar de banco.

Atividade: Clientes inativos têm um risco de churn significativamente maior, independentemente do saldo.

🧠 7. Modelagem de Dados & Machine Learning
7.1 Separação Treino-Teste
Os dados foram divididos em 80% para treino e 20% para teste, mantendo a proporção da variável alvo (estratificação).

7.2 Algoritmo Selecionado: XGBoost Classifier
Optamos pelo XGBoost devido à sua alta performance em problemas de classificação com dados estruturados e sua robustez contra overfitting.

Hiperparâmetros do modelo:

python
{
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss'
}
📉 8. Avaliação do Modelo e Resultados de Negócio
Após o treinamento, o modelo foi avaliado no conjunto de teste, gerando os seguintes resultados:

8.1 Métricas de Performance
Métrica	Valor
Acurácia	86.10%
AUC ROC	0.8613
A AUC de 0.861 demonstra que o modelo tem excelente capacidade de distinguir um cliente que vai sair de um que vai ficar.

8.2 Tradução para o Negócio
Aplicando o modelo a todos os clientes da base de teste, o resultado foi:

Das 2.000 transações analisadas, o modelo identificou 407 clientes com alto risco de evasão.

Top 5 Clientes com Maior Risco:

Age	Balance	CreditScore	Probabilidade_Churn
46	115,248	727	99.5%
55	111,363	547	99.2%
52	139,493	469	99.2%
Insights de Ação:

Foco na Retenção: A equipe de marketing deve priorizar o contato com os clientes que apresentam probabilidade > 90%.

Estratégia Personalizada: Para o perfil de cliente identificado (ex: idade >45, saldo médio/alto, crédito médio), uma ação personalizada (como um gerente de conta dedicado ou isenção de tarifas) pode ser mais eficaz.

Prejuízo Evitado: Considerando um ticket médio hipotético de R
1.000
/
m
e
^
s
p
o
r
c
l
i
e
n
t
e
,
r
e
t
e
r
30
1.000/m 
e
^
 sporcliente,reter30 1.2 milhão** no próximo semestre.

⚙️ 9. Importância das Variáveis
As variáveis mais importantes para o modelo foram:

Complain (Reclamação): A mais influente por uma grande margem. Clientes que registraram reclamações quase certamente irão sair.

NumOfProducts (Nº de Produtos): Clientes com poucos produtos são mais suscetíveis a sair.

Age (Idade): A idade é um forte preditor para este conjunto de dados.

IsActiveMember (Membro Ativo): Inatividade é um forte sinal de alerta.

🚀 10. Como Reproduzir o Projeto
Clone o repositório:

bash
git clone https://github.com/seu-usuario/churn-bank-prediction.git
Instale as dependências:

bash
pip install -r requirements.txt
Execute o Jupyter Notebook:
Navegue até o arquivo Untitled.ipynb e execute as células sequencialmente.

🛠️ 11. Ferramentas Utilizadas
Linguagem: Python 3.8+

Bibliotecas:

pandas, numpy (Manipulação de dados)

matplotlib, seaborn (Visualização)

scikit-learn (Pré-processamento e métricas)

xgboost (Modelo de Machine Learning)

joblib (Persistência do modelo)

📌 12. Conclusão
O projeto atingiu seu objetivo principal, entregando um modelo de Machine Learning (XGBoost) robusto e acionável. Com uma acurácia de 86% e uma AUC de 0.86, o modelo consegue identificar com boa confiança os clientes com maior propensão à evasão.

A principal descoberta de negócio é a confirmação de que clientes que reclamaram têm uma probabilidade altíssima de sair, apontando uma oportunidade imediata de criação de um workflow de "retenção de última milha" no SAC.

Os próximos passos incluem o deploy deste modelo via API para que possa ser consumido por ferramentas de CRM, permitindo a automação das campanhas de retenção.

📋 13. Próximos Passos & Melhorias Futuras
Deploy do Modelo: Criar uma API simples (usando Flask ou FastAPI) para disponibilizar o modelo para outros times.

Novos Ciclos do CRISP: Coletar mais dados (ex: tempo desde a última interação) para refinar as previsões.

Balanceamento de Classes: Aplicar técnicas como SMOTE para lidar com o desbalanceamento da classe "Churn" (apenas ~20%) e potencialmente aumentar a taxa de acerto para os clientes que saem.

Teste de Outros Algoritmos: Avaliar o desempenho do Random Forest e Redes Neurais para comparação.

👥 14. Autor
Seu Nome

LinkedIn: https://linkedin.com/in/túlio-santos-b65720a4/

GitHub: https://github.com/tuliosannttos











