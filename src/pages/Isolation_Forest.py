import streamlit as st
import pandas as pd
import numpy as np


def app():
    st.title('Isolation Forest')

    df_anomalias = pd.read_csv('mining_learning/ensemble/output_isolation_forest_graphs/isolation_forest_anomalous_transactions.csv')
    df_normais = pd.read_csv('mining_learning/ensemble/output_isolation_forest_graphs/isolation_forest_normal_transactions.csv')

    total_transactions = len(df_anomalias) + len(df_normais)
    total_anomalias = len(df_anomalias)
    total_normais = len(df_normais)
    percentual_anomalias = total_anomalias / total_transactions * 100

    st.write('## Introdução')
    st.write("O Isolation Forest foi treinado com 150 estimadores, máximo de amostras equivalente a 10% do tamanho do dataset, contaminação de 0.05 e normalizado com RobustScaler.")

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
    col1.image('mining_learning/ensemble/output_isolation_forest_graphs/feature_correlation_heatmap.png', use_column_width=True)
    col1.write(" ")

    col2.markdown("### Curva ROC")
    col2.image('mining_learning/ensemble/output_isolation_forest_graphs/roc_curve.png', use_column_width=True)

    col1.markdown("### Anomalias detectadas")
    col1.image('mining_learning/ensemble/output_isolation_forest_graphs/anomalies_detection.png', use_column_width=True)

    col2.markdown("### Histograma de Scoores de Anomalias")
    col2.image('mining_learning/ensemble/output_isolation_forest_graphs/anomaly_scores_histogram.png', use_column_width=True)

    col1.markdown("### Densidade dos scores de anomalias")
    col1.image('mining_learning/ensemble/output_isolation_forest_graphs/density_anomaly_scores.png', use_column_width=True)

    col2.markdown("### Bloxplot dos scores de anomalias")
    col2.image('mining_learning/ensemble/output_isolation_forest_graphs/boxplot_anomaly_scores.png', use_column_width=True)

    st.write('## Scores de anomalias vs Features')

    col3, col4 = st.columns(2)

    col3.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_eth_close_price_1h.png', use_column_width=True)
    col4.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_eth_price_change_1h.png', use_column_width=True)

    col3.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_eth_volume_1h.png', use_column_width=True)
    col4.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_gas_limit_in_block.png', use_column_width=True)

    col3.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_gas_price.png', use_column_width=True)
    col4.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_gas_used_by_block.png', use_column_width=True)

    col3.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_gas_used_by_txn.png', use_column_width=True)
    col4.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_max_fee_per_gas.png', use_column_width=True)

    col3.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_max_priority_fee_per_gas.png', use_column_width=True)
    col4.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_txn_fee.png', use_column_width=True)

    col3.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_txn_in_block_count.png', use_column_width=True)
    col4.image('mining_learning/ensemble/output_isolation_forest_graphs/scatter_anomalia_value_usd.png', use_column_width=True)


if __name__ == '__main__':
    app()