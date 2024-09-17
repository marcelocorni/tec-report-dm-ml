import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc
from autoEncoder import Autoencoder
import time


# Adicionar o diretório principal do projeto no sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import lib.databaseService as db_service


start_time = time.time()

# Configurar o diretório para salvar os resultados
output_dir = 'output_auto_encoder_graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Carregar dados do banco de dados
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



# Selecionar as colunas numéricas e normalizar os dados com RobustScaler
features = [
    'txn_in_block_count', 'eth_close_price_1h', 'eth_price_change_1h',
    'eth_volume_1h', 'max_fee_per_gas', 'max_priority_fee_per_gas', 'value_usd',
    'txn_fee', 'gas_price', 'gas_limit_in_block', 'gas_used_by_txn', 'gas_used_by_block',
]

# Armazenar 'blockno' e 'transaction_hash' antes da normalização
blockno_transaction_hash = df[['blockno', 'transaction_hash']]

# Normalizar os dados
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# Definir o Autoencoder com 12 features de entrada e 4 dimensões no espaço latente
input_dim = 12
latent_space_dim = 4
learning_rate = 0.05
batch_size = 32
epochs = 20

# Dividir os dados em treino e teste
X_train, X_test, blockno_hash_train, blockno_hash_test = train_test_split(df_scaled, blockno_transaction_hash, test_size=0.3, random_state=42)

# Criar o Autoencoder simplificado
autoencoder = Autoencoder(input_dim, latent_space_dim, learning_rate)

# Treinar o Autoencoder
history = autoencoder.train(X_train, batch_size=batch_size, epochs=epochs, validation_data=X_test)

# Obter a representação latente dos dados de treino
latent_representation = autoencoder.encode(X_train)

# Reconstruir os dados a partir da representação latente
reconstructed_data = autoencoder.decode(latent_representation)

# --- Detecção de Anomalias ---
reconstruction_errors = np.mean(np.abs(X_train - reconstructed_data), axis=1)

# Verificar e remover valores NaN dos erros de reconstrução
if np.isnan(reconstruction_errors).any():
    print("Valores NaN encontrados em reconstruction_errors. Removendo...")
    reconstruction_errors = reconstruction_errors[~np.isnan(reconstruction_errors)]

# Definir um limiar para anomalias (percentil 95)
threshold = np.percentile(reconstruction_errors, 95)
anomalies = reconstruction_errors > threshold

# Associando as anomalias aos dados originais
df_train = pd.DataFrame(X_train, columns=features)
df_train['blockno'] = blockno_hash_train['blockno'].values
df_train['transaction_hash'] = blockno_hash_train['transaction_hash'].values
df_train['reconstruction_error'] = reconstruction_errors
df_train['anomaly'] = anomalies

# --- Salvando os Resultados em CSV ---
df_anomalies = df_train[df_train['anomaly'] == True][['blockno', 'transaction_hash', 'reconstruction_error','anomaly']]
df_anomalies.to_csv(os.path.join(output_dir, 'auto_encoder_anomalous_transactions.csv'), index=False)

df_normals = df_train[df_train['anomaly'] == False][['blockno', 'transaction_hash', 'reconstruction_error','anomaly']]
df_normals.to_csv(os.path.join(output_dir, 'auto_encoder_normal_transactions.csv'), index=False)

# --- Visualização Gráfica ---


# --- Gráfico de Treino vs Validação ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda de Treinamento vs Perda de Validação no Autoencoder')
plt.xlabel('Épocas')
plt.ylabel('Perda (Loss)')
plt.legend()
plt.grid(True)
# Salvar o gráfico
plt.savefig(os.path.join(output_dir, 'train_val_loss_autoencoder.png'))
plt.close()

# 1. Gráfico de Erros de Reconstrução
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_errors, bins=50, alpha=0.75, color='blue')
plt.axvline(threshold, color='red', linestyle='--', label='Limiar para Anomalias')
plt.title('Erros de Reconstrução')
plt.xlabel('Erro de Reconstrução')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'reconstruction_errors_histogram.png'))
plt.close()

# 2. Gráfico de Detecção de Anomalias
plt.figure(figsize=(10, 6))
plt.scatter(range(len(reconstruction_errors)), reconstruction_errors, c=anomalies, cmap='coolwarm')
plt.axhline(threshold, color='red', linestyle='--', label='Limiar para Anomalias')
plt.title('Detecção de Anomalias com Base no Erro de Reconstrução')
plt.xlabel('Índice da Transação')
plt.ylabel('Erro de Reconstrução')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'anomalies_detection.png'))
plt.close()

# 3. Boxplot dos erros de reconstrução
plt.figure(figsize=(10, 6))
plt.boxplot(reconstruction_errors, vert=False)
plt.title('Boxplot de Erros de Reconstrução')
plt.xlabel('Erro de Reconstrução')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'boxplot_reconstruction_errors.png'))
plt.close()

# --- Calcula a matriz de correlação ---
corr_matrix = pd.DataFrame(df_scaled, columns=features).corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de Calor das Correlações entre as Features')
plt.xticks(rotation=45, ha='right')  # Girar os nomes das features no eixo X
plt.yticks(rotation=0)  # Garantir que as features no eixo Y fiquem na horizontal
plt.tight_layout()  # Ajustar o layout para evitar cortes
plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'))
plt.close()


# 5. Scatter plot dos erros de anomalia por feature
for feature in features:
    plt.figure(figsize=(10, 6))
    plt.scatter(df_train[feature], reconstruction_errors, c=anomalies, cmap='coolwarm')
    plt.title(f'Erros de Reconstrução vs {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Erro de Reconstrução')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'scatter_reconstrucao_{feature}.png'))
    plt.close()

# --- Densidade dos erros de reconstrução ---
plt.figure(figsize=(10, 6))
sns.kdeplot(reconstruction_errors, fill=True, color="b")
plt.title('Densidade dos Erros de Reconstrução')
plt.xlabel('Erro de Reconstrução')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'density_reconstruction_errors.png'))
plt.close()


# Definir os rótulos verdadeiros (exemplo: 0 para normais, 1 para anomalias)
# Aqui estou assumindo que você já tem um critério para definir os rótulos (exemplo fictício abaixo)
# Em um caso real, você precisaria ter dados rotulados ou definir rótulos de maneira mais robusta.
true_labels = np.zeros(len(reconstruction_errors))  # Suponha que inicialmente todos são normais
true_labels[reconstruction_errors > threshold] = 1  # Classificar como anomalias as amostras com erro acima do limiar

# Calcular a curva ROC
fpr, tpr, thresholds = roc_curve(true_labels, reconstruction_errors)
roc_auc = auc(fpr, tpr)

# Plotar a curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva Característica de Operação do Receptor')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'autoencoder_roc_curve.png'))
plt.close()

# Exibir o tempo de execução
end_time = time.time()
execution_time = end_time - start_time
hours, rem = divmod(execution_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Tempo de execução: {int(hours)} horas, {int(minutes)} minutos e {int(seconds)} segundos para processar {len(df)} transações.")
