import streamlit as st
import pandas as pd
import numpy as np


def app():
    st.title('Auto Encoder')

    df_anomalias = pd.read_csv('mining_learning/tensorKeras/output_auto_encoder_graphs/auto_encoder_anomalous_transactions.csv')
    df_normais = pd.read_csv('mining_learning/tensorKeras/output_auto_encoder_graphs/auto_encoder_normal_transactions.csv')

    total_transactions = len(df_anomalias) + len(df_normais)
    total_anomalias = len(df_anomalias)
    total_normais = len(df_normais)
    percentual_anomalias = total_anomalias / total_transactions * 100

    st.write('## Introdução')
    st.write("O Autoencoder foi treinado com 20 épocas e 4 dimensões no espaço latente, com taxa de aprendizado de 0.05 e tamanho de lote de 32 e 12 features de entrada, normalizadas com MinMaxScaler.")
    
    st.write('## Anomalias')
    st.write(df_anomalias.head())

    st.write('## Normais')
    st.write(df_normais.head())


    st.write('## Resultados')
    
    st.write(f"Total de transações: `{total_transactions}`")
    st.write(f"Total de anomalias: `{total_anomalias}` (`{percentual_anomalias:.2f}%`)")
    st.write(f"Total de transações normais: `{total_normais}`")
    
    col1, col2 = st.columns(2)

    col1.markdown("### Matriz de Correlação")
    col1.image('mining_learning/tensorKeras/output_auto_encoder_graphs/feature_correlation_heatmap.png', use_column_width=True)
    col1.write(" ")

    col2.markdown("### Curva ROC")
    col2.image('mining_learning/tensorKeras/output_auto_encoder_graphs/autoencoder_roc_curve.png', use_column_width=True)

    col1.markdown("### Anomalias detectadas")
    
    col1.image('mining_learning/tensorKeras/output_auto_encoder_graphs/anomalies_detection.png', use_column_width=True)

    col2.markdown("### Treino e validação do Autoencoder")
    
    col2.image('mining_learning/tensorKeras/output_auto_encoder_graphs/train_val_loss_autoencoder.png', use_column_width=True)

    col1.markdown("### Densidade dos erros de reconstrução")

    col1.image('mining_learning/tensorKeras/output_auto_encoder_graphs/density_reconstruction_errors.png', use_column_width=True)

    col2.markdown("### Bloxplot dos erros de reconstrução")

    col2.image('mining_learning/tensorKeras/output_auto_encoder_graphs/boxplot_reconstruction_errors.png', use_column_width=True)

    st.write('## Erros de reconstrução vs Features')

    col3, col4 = st.columns(2)

    col3.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_eth_close_price_1h.png', use_column_width=True)
    col4.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_eth_price_change_1h.png', use_column_width=True)

    col3.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_eth_volume_1h.png', use_column_width=True)
    col4.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_gas_limit_in_block.png', use_column_width=True)

    col3.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_gas_price.png', use_column_width=True)
    col4.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_gas_used_by_block.png', use_column_width=True)

    col3.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_gas_used_by_txn.png', use_column_width=True)
    col4.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_max_fee_per_gas.png', use_column_width=True)

    col3.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_max_priority_fee_per_gas.png', use_column_width=True)
    col4.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_txn_fee.png', use_column_width=True)

    col3.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_txn_in_block_count.png', use_column_width=True)
    col4.image('mining_learning/tensorKeras/output_auto_encoder_graphs/scatter_reconstrucao_value_usd.png', use_column_width=True)


if __name__ == '__main__':
    app()
