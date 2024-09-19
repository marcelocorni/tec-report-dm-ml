from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.metrics import roc_auc_score, roc_curve, auc
import time

# Adicionar o diretório principal do projeto no sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import lib.databaseService as db_service

# iniciar o tempo de processamento do script
start_time = time.time()


# Carregar dados
fields = """
        txn_in_block_count, eth_close_price_1h, eth_price_change_1h,
        eth_volume_1h, max_fee_per_gas, max_priority_fee_per_gas, value_usd,
        txn_fee, gas_price, gas_limit_in_block, gas_used_by_txn, gas_used_by_block,
        blockno, transaction_hash
        """

where_clause = "WHERE Status = 'Success'"
orderby = "blockno, transaction_hash"
limit = 0

df = db_service.load_transactions_into_dataframe(fields=fields, where_clause=where_clause, orderby=orderby, limit=limit)

# Remover valores nulos
#df.dropna(inplace=True)

# Preencher valores nulos com a moda
df.fillna(df.mode().iloc[0], inplace=True)

print("Total de transações carregadas:", len(df))

# --- Pré-processamento ---
# Selecionando as colunas numéricas e normalizando os dados com RobustScaler
features = [
    'txn_in_block_count', 'eth_close_price_1h', 'eth_price_change_1h',
    'eth_volume_1h', 'max_fee_per_gas', 'max_priority_fee_per_gas', 'value_usd',
    'txn_fee', 'gas_price', 'gas_limit_in_block', 'gas_used_by_txn', 'gas_used_by_block',
]


# print("Realizando a imputação dos dados faltantes...")
# Inicializando o imputador KNN com n_neighbors = 5
# imputer = KNNImputer(n_neighbors=5)

# Aplicando o KNNImputer aos dados
# df[features] = imputer.fit_transform(df[features])


# Armazenar 'blockno' e 'transaction_hash' antes da normalização
blockno_transaction_hash = df[['blockno', 'transaction_hash']]

print("Normalizando os dados...")

scaler = RobustScaler()
df_scaled = scaler.fit_transform(df[features])

print("Dividindo os dados em treino e teste...")

# --- Train Test Split ---
# Dividir o conjunto de dados em treino (70%) e teste (30%)
# --- Train Test Split ---
X_train, X_test, blockno_hash_train, blockno_hash_test = train_test_split(df_scaled, blockno_transaction_hash, test_size=0.3, random_state=42)

# Função de avaliação para o Bayesian Optimization
def isolation_forest_evaluate(n_estimators, max_samples, contamination):
    n_estimators = int(n_estimators)
    max_samples = int(max_samples)
    contamination = float(contamination)
    
    # Treinando o modelo Isolation Forest
    model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                            contamination=contamination, random_state=42)
    model.fit(X_train)
    
    # Previsões no conjunto de teste
    predictions = model.predict(X_test)
    anomaly_scores = model.decision_function(X_test)
    
    # Assumindo que a maioria dos dados são normais (1) e as anomalias (-1)
    y_true = np.ones(len(X_test))
    y_true[predictions == -1] = -1
    
    # AUC como métrica de otimização
    auc_score = roc_auc_score(y_true, anomaly_scores)
    
    return auc_score

# Definindo o intervalo de parâmetros para otimização
param_bounds = {
    'n_estimators': (50, 300),  # Número de estimadores (árvores)
    'max_samples': (100, len(X_train)),  # Número máximo de amostras
    'contamination': (0.01, 0.2)  # Proporção de anomalias
}

# Otimizador Bayesiano
optimizer = BayesianOptimization(
    f=isolation_forest_evaluate,
    pbounds=param_bounds,
    random_state=42
)

# Realizando a otimização
optimizer.maximize(
    init_points=5,  # Número de pontos para explorar aleatoriamente antes da otimização
    n_iter=25  # Número de iterações da otimização
)

# Exibindo os melhores parâmetros
print("Melhores parâmetros encontrados:", optimizer.max)

print("Treinando o modelo...")

# --- Treinamento do Modelo ---
max_samples = 1.0 # Definir o número máximo de amostras para 10% do total
n_estimators = 233  # Número de árvores na floresta
contamination = 0.001  # Proporção de anomalias no conjunto de dados

# Treinar o modelo com os melhores parâmetros encontrados no conjunto de treino
isolation_forest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                   contamination=contamination, random_state=42)
isolation_forest.fit(X_train)

# Previsão e Detecção de Anomalias no conjunto de teste
predictions = isolation_forest.predict(X_test)
anomaly_scores = isolation_forest.decision_function(X_test)

# Associando os Resultados aos Dados Originais
df_test = pd.DataFrame(X_test, columns=features)
df_test['anomaly_score'] = anomaly_scores
df_test['anomaly'] = predictions == -1  # True para anomalias, False para normais

print("Gerando gráficos e salvando os resultados...")

# --- Salvando os Resultados ---

# --- Visualização Gráfica dos Resultados ---
output_dir = 'output_isolation_forest_graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mean_score = np.mean(anomaly_scores)
std_score = np.std(anomaly_scores)
threshold = np.percentile(anomaly_scores, 5)
df_resultados = pd.DataFrame({
    'mean_score': [mean_score],
    'std_score': [std_score],
    'threshold': [threshold]
})
df_resultados.to_csv(os.path.join(output_dir,'isolation_forest_results.csv'), index=False)


# Adicionar 'blockno' e 'transaction_hash' de volta aos DataFrames
df_test = pd.concat([df_test, blockno_hash_test.reset_index(drop=True)], axis=1)

# Salvamento dos Resultados em CSVs
df_anomalies = df_test[df_test['anomaly'] == True][['blockno', 'transaction_hash', 'anomaly_score','anomaly']]
df_anomalies.to_csv(os.path.join(output_dir,'isolation_forest_anomalous_transactions.csv'), index=False)

df_normals = df_test[df_test['anomaly'] == False][['blockno', 'transaction_hash', 'anomaly_score','anomaly']]
df_normals.to_csv(os.path.join(output_dir,'isolation_forest_normal_transactions.csv'), index=False)



# Gráficos, conforme descrito anteriormente

# 1. Histograma dos scores de anomalias
plt.figure(figsize=(10, 6))
plt.hist(anomaly_scores, bins=50, alpha=0.75, color='blue')
plt.axvline(np.percentile(anomaly_scores, 5), color='red', linestyle='--', label='Threshold for Anomalies')
plt.title('Scores de Anomalia')
plt.xlabel('Score de Anomalia')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'anomaly_scores_histogram.png'))
plt.close()

# 2. Gráfico de detecção de anomalias
plt.figure(figsize=(10, 6))
plt.scatter(range(len(anomaly_scores)), anomaly_scores, c=(predictions == -1), cmap='coolwarm')
plt.axhline(np.percentile(anomaly_scores, 5), color='red', linestyle='--', label='Threshold for Anomalies')
plt.title('Detecção de Anomalias')
plt.xlabel('Índice da Transação')
plt.ylabel('Score de Anomalia')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'anomalies_detection.png'))
plt.close()

# --- Gráficos adicionais ---

# 3. Gráfico de curva ROC (Receiver Operating Characteristic) (se tiver rótulos)
# Para o Isolation Forest, você pode gerar uma curva ROC se tiver rótulos de verdade (não anomalias detectadas).

# Simulação do y_true para o conjunto de teste
# No Isolation Forest, assumimos que a maioria dos dados são normais (rotulados como 1)
# e as anomalias como -1, então podemos simular isso se não tivermos os rótulos verdadeiros
y_true = np.ones(len(X_test))  # Assumimos que todos são normais inicialmente
y_true[predictions == -1] = -1  # Marcamos as previsões como -1 (anomalias)

# Agora, calculamos o ROC usando o conjunto de teste e o anomaly_score correspondente
fpr, tpr, thresholds = roc_curve(y_true, anomaly_scores)
roc_auc = auc(fpr, tpr)

# Exibir a AUC
print(f"AUC: {roc_auc}")

# --- Gráfico da curva ROC ---
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva Característica de Operação do Receptor')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
plt.close()

# 4. Boxplot dos scores de anomalias
plt.figure(figsize=(10, 6))
plt.boxplot(anomaly_scores, vert=False)
plt.title('Boxplot de Scores de Anomalia')
plt.xlabel('Score de Anomalia')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'boxplot_anomaly_scores.png'))
plt.close()

# 5. Scatter plot dos erros de anomalia por feature
for feature in features:
    plt.figure(figsize=(10, 6))
    plt.scatter(df_test[feature], df_test['anomaly_score'], c=(df_test['anomaly']), cmap='coolwarm')
    plt.title(f'Scores de Anomalia vs {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Score de Anomalia')
    plt.colorbar(label='Anomalia')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'scatter_anomalia_{feature}.png'))
    plt.close()

# 6. Gráfico de densidade dos scores de anomalias
plt.figure(figsize=(10, 6))
sns.kdeplot(anomaly_scores, fill=True, color="b")
plt.title('Densidade dos Scores de Anomalia')
plt.xlabel('Score de Anomalia')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'density_anomaly_scores.png'))
plt.close()


# Calcula a matriz de correlação
corr_matrix = pd.DataFrame(df_scaled, columns=features).corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor das Correlações entre as Features')
plt.xticks(rotation=45, ha='right')  # Girar os nomes das features no eixo X
plt.yticks(rotation=0)  # Garantir que as features no eixo Y fiquem na horizontal
plt.tight_layout()  # Ajustar o layout para evitar cortes
plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'))
plt.close()

# Exibir o tempo de execução em Horas, Minutos e Segundos e o Total de Transações Carregadas
end_time = time.time()
execution_time = end_time - start_time
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Tempo de execução: {int(hours)} horas, {int(minutes)} minutos e {int(seconds)} segundos para processar {len(df)} transações.")