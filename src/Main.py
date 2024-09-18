import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
import lib.databaseService as BancoDeDados

def main():
    st.set_page_config(page_title='Relatório', page_icon=':chart_with_upwards_trend:', layout='wide')

    st.title('Detecção de Ataques de Front-Running na Blockchain Ethereum aplicando técnicas de Mineração de Dados e Aprendizado de Máquina')

    st.markdown('## Introdução')
    st.markdown('Foi desenvolvido um Pipeline para a detecção de ataques de Front-Running.')

    st.image('../images/pipeline.png', use_column_width=True)

    st.markdown('## Análise Exploratória de Dados')
    st.markdown('Nesta seção, iremos realizar uma análise exploratória de dados para entender melhor o dataset que estamos trabalhando.')

    col1, col2, col3 = st.columns(3)
    # Carregar os dados em um DataFrame para navegar pelas páginas
    limit = 10000
    page = 0
    total_pages = 10000
    page = col1.number_input('Página', min_value=0, max_value=total_pages, value=page)
    limit = col2.number_input('Tamanho da Página', min_value=1000, max_value=10000, value=limit)
    with st.spinner(f'Carregando dados da página `{page}`...'):
        total_pages, total_records, df = BancoDeDados.load_transactions_into_dataframe_paged(page, limit)
    col3.markdown(f'###### Total de registros: `{total_records}`')
    col3.markdown(f'###### Total de páginas: `{total_pages}`')
    st.write(df)

    # Plotar gráfico de barras com a quantidade de transações por dia
    st.markdown('### Quantidade de Transações por Dia')
    df_grouped_by_date = BancoDeDados.load_transactions_into_dataframe_grouped_by_date()
    colormap = px.colors.cyclical.Phase_r
    
    fig = px.bar(df_grouped_by_date, x='data', y='total', title='', color='data', color_discrete_sequence=colormap)
    # Colocar a cor de todos os textos do gráfico para preto para melhor visualização em fundo branco
    # atualizar o formato da data no eixo x
    fig.update_xaxes(type='category')
    # trocar os labels dos exios x e y
    fig.update_layout(xaxis_title='Data', yaxis_title='Total de Transações')
    # Exibir os valores no eixo y
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    # Inserir linha de referência na média
    fig.add_hline(y=df_grouped_by_date['total'].mean(), line_dash='dot', line_color='gray', annotation_text='', annotation_position='right right')

    st.plotly_chart(fig)


    # Plotar gráfico de barras com a soma do valor em USD por transação por dia
    st.markdown('### Valor em USD por Transação por Dia')
    df_grouped_by_date = BancoDeDados.load_sum_value_usd_by_date()
    colormap = px.colors.cyclical.Phase_r
    fig = px.bar(df_grouped_by_date, x='transaction_date', y='value_usd', title='', color='transaction_date', color_discrete_sequence=colormap)
    # Colocar a cor de todos os textos do gráfico para preto para melhor visualização em fundo branco
    # atualizar o formato da data no eixo x
    fig.update_xaxes(type='category')
    # trocar os labels dos exios x e y
    fig.update_layout(xaxis_title='Data', yaxis_title='Soma do Valor em USD')
    # Exibir os valores no eixo y
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    # Inserir linha de referência na média
    fig.add_hline(y=df_grouped_by_date['value_usd'].mean(), line_dash='dot', line_color='gray', annotation_text='', annotation_position='right right')
    st.plotly_chart(fig)


    # Plotar gráfico de barras com a quantidade de transações por hora do dia
    st.markdown('### Quantidade de Transações por Hora do Dia')
    df_grouped_by_hour = BancoDeDados.load_transactions_into_dataframe_grouped_by_hour()
    df_grouped_by_hour["hora"] = df_grouped_by_hour["hora"].astype(str)
    colormap = px.colors.cyclical.Phase_r
    fig = px.bar(df_grouped_by_hour, x='hora', y='total', title='', color='hora', color_discrete_sequence=colormap)
    # atualizar o formato da hora no eixo x
    fig.update_xaxes(type='category')
    # trocar os labels dos exios x e y
    fig.update_layout(xaxis_title='Hora', yaxis_title='Total de Transações')
    # Exibir os valores no eixo y
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    # Inserir linha de referência na média
    fig.add_hline(y=df_grouped_by_hour['total'].mean(), line_dash='dot', line_color='gray', annotation_text='', annotation_position='left left')
    # Colocar a cor de todos os textos do gráfico para preto para melhor visualização em fundo branco
    st.plotly_chart(fig)

    # Plotar gráfico combinado com o preço de fechamento do ETH em USD por dia e o volume máximo em USD do ETH por dia
    st.markdown('### Feeatures extraídas da API da Binance (dados de Candlesticks)')
    df_eth_price_usd = BancoDeDados.load_eth_price_usd_by_date()
    df_eth_volume_usd = BancoDeDados.load_max_volume_usd_eth_by_date()
    
    #Plotar dois subplots em colunas com o preço de fechamento do ETH em USD por dia e o volume máximo em USD do ETH por dia
    fig = px.line(df_eth_price_usd, x='hora', y='preco', title='Preço de Fechamento do ETH em USD por Dia')
    fig2 = px.line(df_eth_volume_usd, x='hora', y='volume', title='Volume Máximo em USD do ETH por Dia')
    fig.update_xaxes(type='category')
    fig2.update_xaxes(type='category')
    fig.update_layout(xaxis_title='Hora', yaxis_title='Preço de Fechamento do ETH em USD')
    fig2.update_layout(xaxis_title='Hora', yaxis_title='Volume Máximo em USD do ETH')
    fig.add_hline(y=df_eth_price_usd['preco'].mean(), line_dash='dot', line_color='gray', annotation_text='', annotation_position='left left')
    fig2.add_hline(y=df_eth_volume_usd['volume'].mean(), line_dash='dot', line_color='gray', annotation_text='', annotation_position='left left')
    # Colocar a cor de todos os textos do gráfico para preto para melhor visualização em fundo branco
    st.plotly_chart(fig)
    st.plotly_chart(fig2)

    # Protótipo do sistema de classificação de Front-Running
    st.markdown('## Protótipo do Sistema de Classificação de Front-Running')
    st.image('../images/classificador.png', use_column_width=True)
   
if __name__ == '__main__':
    main()