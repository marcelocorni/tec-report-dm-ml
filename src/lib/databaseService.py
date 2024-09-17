import pandas as pd
import streamlit as st
import sqlalchemy as sa
from sqlalchemy import text

# Configura a string de conexão
DATABASE_URI = "postgresql://postgres:123456@localhost:65432/tx18908894_18958621"

# Cria o engine com o SQLAlchemy
engine = sa.create_engine(DATABASE_URI)

# Função para conectar ao banco de dados PostgreSQL usando SQLAlchemy
def connect_to_db():
    try:
        return engine
    except Exception as e:
        print(f"Erro ao se conectar ao banco de dados: {e}")
        return None

# Função para carregar os dados paginados em um DataFrame retornando o total de páginas, o total de registros e o DataFrame
def load_transactions_into_dataframe_paged(page, limit):
    engine = connect_to_db()
    
    if engine is None:
        return None, None, None
    
    try:
        # Carregar dados da tabela transactions em um DataFrame do pandas
        query = text(f"SELECT * FROM transactions ORDER BY blockno, id OFFSET {page * limit} LIMIT {limit};")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        # Contar o total de registros
        count_query = text("SELECT COUNT(*) FROM transactions;")
        with engine.connect() as conn:
            total_records = pd.read_sql(count_query, conn).values[0][0]
        
        # Calcular o total de páginas
        total_pages = total_records // limit
        
        print("Dados carregados com sucesso!")
        return total_pages, total_records, df
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None, None, None

# Função para retornar um dataFrame com a quantidade de registros por Data (YYYY-MM-DD)
def load_transactions_into_dataframe_grouped_by_date():
    engine = connect_to_db()
    
    if engine is None:
        return None
    
    try:
        # Carregar dados da tabela transactions em um DataFrame do pandas
        query = text("SELECT DATE(date_time_utc) as data, COUNT(*) as total FROM transactions GROUP BY DATE(date_time_utc) ORDER BY data;")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print("Dados carregados com sucesso!")
        return df
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

# Função para retornar um dataFrame com a quantidade de registros por hora
def load_transactions_into_dataframe_grouped_by_hour():
    engine = connect_to_db()
    
    if engine is None:
        return None
    
    try:
        # Carregar dados da tabela transactions em um DataFrame do pandas
        query = text("SELECT transaction_hour as hora, COUNT(*) as total FROM transactions GROUP BY 1 ORDER BY 1;")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print("Dados carregados com sucesso!")
        return df
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

# Função para carregar o preço de fechamento do ETH em USD por dia
def load_eth_price_usd_by_date():
    engine = connect_to_db()
    
    if engine is None:
        return None
    
    try:
        # Carregar dados da tabela transactions em um DataFrame do pandas
        query = text("SELECT transaction_hour as hora, MAX(eth_close_price_1h) as preco FROM transactions GROUP BY transaction_hour ORDER BY hora;")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print("Dados carregados com sucesso!")
        return df
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

# Função para carregar o volume máximo em USD do ETH por dia
def load_max_volume_usd_eth_by_date():
    engine = connect_to_db()
    
    if engine is None:
        return None
    
    try:
        # Carregar dados da tabela transactions em um DataFrame do pandas
        query = text("SELECT transaction_hour as hora, SUM(eth_volume_1h) as volume FROM transactions GROUP BY transaction_hour ORDER BY transaction_hour;")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print("Dados carregados com sucesso!")
        return df
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None

# Função para carregar os dados em um DataFrame
def load_transactions_into_dataframe(fields="*",where_clause="",orderby="blockno, id",limit=10000):
    engine = connect_to_db()
    
    if engine is None:
        return None
    
    try:
        # Carregar dados da tabela transactions em um DataFrame do pandas
        query = text(f"SELECT {fields} FROM transactions {where_clause} ORDER BY {orderby} {"" if limit == 0 else f"LIMIT {limit}"};")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print("Dados carregados com sucesso!")
        return df
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None


# Funcção para retornar soma de value_usd por transação por dia
def load_sum_value_usd_by_date():
    engine = connect_to_db()
    
    if engine is None:
        return None
    
    try:
        # Carregar dados da tabela transactions em um DataFrame do pandas
        query = text("SELECT  DATE(date_time_utc) AS transaction_date,  SUM(value_usd) AS value_usd FROM  public.transactions GROUP BY  DATE(date_time_utc) ORDER BY  transaction_date;")
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        print("Dados carregados com sucesso!")
        return df
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return None





