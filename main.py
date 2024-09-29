import yfinance as yf
import pandas as pd
import datetime as dt
import schedule
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configurações e Constantes
acao1 = input('Digite a primeira Ação: ')
acao2 = input('Digite a segunda Ação: ')
TICKER_NASDAQ_1 = acao1
TICKER_NASDAQ_2 = acao2
START_DATE = '2024-01-01'
END_DATE = dt.datetime.now().strftime('%Y-%m-%d')

# Configuração de logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_data(ticker: str) -> pd.DataFrame:
    """
    Busca dados históricos de uma ação.

    Args:
        ticker (str): O símbolo da ação.

    Returns:
        pd.DataFrame: DataFrame contendo os dados históricos da ação.
    """
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE)
        if data.empty:
            logging.warning(f"Nenhum dado encontrado para o ticker: {ticker}")
        return data
    except Exception as e:
        logging.error(f"Erro ao buscar dados para {ticker}: {e}")
        return pd.DataFrame()


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula indicadores técnicos para os dados de uma ação.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados da ação.

    Returns:
        pd.DataFrame: DataFrame com os indicadores calculados.
    """
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()
    data['Volatility'] = data['Close'].rolling(window=30).std()
    data['RSI'] = calculate_rsi(data, window=14)
    data['MACD'], data['Signal'], data['MACD_Hist'] = calculate_macd(
        data['Close'])
    return data


def calculate_rsi(data: pd.DataFrame, window: int) -> pd.Series:
    """
    Calcula o Índice de Força Relativa (RSI).

    Args:
        data (pd.DataFrame): DataFrame contendo os dados da ação.
        window (int): Período para cálculo do RSI.

    Returns:
        pd.Series: Série contendo os valores do RSI.
    """
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calcula o MACD (Moving Average Convergence Divergence).

    Args:
        close (pd.Series): Série de preços de fechamento.
        fast (int): Período da média móvel rápida.
        slow (int): Período da média móvel lenta.
        signal (int): Período da linha de sinal.

    Returns:
        tuple: MACD, linha de sinal e histograma.
    """
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def plot_data(data: pd.DataFrame, ticker: str) -> None:
    """
    Plota os dados da ação com suas médias móveis, volatilidade, RSI e MACD usando Plotly.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados da ação e indicadores.
        ticker (str): O símbolo da ação.
    """
    if data.empty:
        logging.warning(f"Nenhum dado para plotar para {ticker}.")
        return

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=('Preço e Médias Móveis', 'Volatilidade', 'RSI', 'MACD'))

    # Gráfico de Preço e Médias Móveis
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Preço de Fechamento', line=dict(
        color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], name='Média Móvel de 50 Dias', line=dict(
        color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], name='Média Móvel de 200 Dias', line=dict(
        color='green')), row=1, col=1)

    # Gráfico de Volatilidade
    fig.add_trace(go.Scatter(x=data.index, y=data['Volatility'], name='Volatilidade (30 dias)', line=dict(
        color='red')), row=2, col=1)

    # Gráfico de RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(
        color='purple')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # Gráfico de MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(
        color='blue')), row=4, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], name='Linha de Sinal', line=dict(
        color='red')), row=4, col=1)
    fig.add_trace(go.Bar(
        x=data.index, y=data['MACD_Hist'], name='Histograma MACD', marker_color='gray'), row=4, col=1)

    fig.update_layout(height=1200, width=1000,
                      title_text=f"Análise Técnica da Ação {ticker}")
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Volatilidade", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    fig.show()


def save_to_csv(data: pd.DataFrame, ticker: str) -> None:
    """
    Salva os dados da ação em um arquivo CSV.

    Args:
        data (pd.DataFrame): DataFrame contendo os dados da ação.
        ticker (str): O símbolo da ação.
    """
    filename = f"{ticker}_dados.csv"
    data.to_csv(filename)
    logging.info(f"Dados salvos em {filename}")


def analyze_stock(ticker: str) -> None:
    """
    Realiza a análise de ações e salva os dados em CSV.

    Args:
        ticker (str): O símbolo da ação a ser analisada.
    """
    data = fetch_data(ticker)
    if data.empty:
        logging.warning(f"Nenhum dado encontrado para {ticker}.")
        return

    data = calculate_indicators(data)
    plot_data(data, ticker)
    save_to_csv(data, ticker)

    # Análise adicional
    last_close = data['Close'].iloc[-1]
    last_sma50 = data['SMA50'].iloc[-1]
    last_sma200 = data['SMA200'].iloc[-1]
    last_rsi = data['RSI'].iloc[-1]
    last_macd = data['MACD'].iloc[-1]
    last_signal = data['Signal'].iloc[-1]

    logging.info(f"Análise para {ticker}:")
    logging.info(f"Último preço de fechamento: {last_close:.2f}")
    logging.info(f"SMA50: {last_sma50:.2f}, SMA200: {last_sma200:.2f}")
    logging.info(f"RSI: {last_rsi:.2f}")
    logging.info(f"MACD: {last_macd:.2f}, Sinal: {last_signal:.2f}")

    if last_close > last_sma50 > last_sma200:
        logging.info("Tendência de alta: Preço acima das médias móveis.")
    elif last_close < last_sma50 < last_sma200:
        logging.info("Tendência de baixa: Preço abaixo das médias móveis.")
    else:
        logging.info("Tendência indefinida.")

    if last_rsi > 70:
        logging.info("RSI indica sobrecompra.")
    elif last_rsi < 30:
        logging.info("RSI indica sobrevenda.")

    if last_macd > last_signal:
        logging.info("MACD acima da linha de sinal: Possível sinal de compra.")
    elif last_macd < last_signal:
        logging.info("MACD abaixo da linha de sinal: Possível sinal de venda.")


def job() -> None:
    """Executa a tarefa programada."""
    analyze_stock(TICKER_NASDAQ_1)
    analyze_stock(TICKER_NASDAQ_2)


def schedule_jobs() -> None:
    """Agenda a execução diária das tarefas."""
    now = (dt.datetime.now() + dt.timedelta(seconds=3)).strftime('%H:%M:%S')
    schedule.every().day.at(now).do(job)
    logging.info(f"Tarefa agendada para {now}")

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    schedule_jobs()
