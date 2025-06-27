#!/usr/bin/env python3
"""
Script de Pré-processamento e Engenharia de Features
===================================================

Este script processa os dados brutos coletados e adiciona indicadores técnicos
necessários para a análise de Machine Learning.

Indicadores calculados:
- Retorno Percentual (1 período)
- Médias Móveis Simples (SMA 20 e 50)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Volume Delta (simplificado)
- Bollinger Bands
- Stochastic Oscillator
- ATR (Average True Range)
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessamento.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calcular_rsi(precos, periodo=14):
    """
    Calcula o RSI (Relative Strength Index).
    
    Args:
        precos (Series): Série de preços
        periodo (int): Período para cálculo do RSI
    
    Returns:
        Series: Valores do RSI
    """
    delta = precos.diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
    rs = ganho / perda
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_macd(precos, rapida=12, lenta=26, sinal=9):
    """
    Calcula o MACD (Moving Average Convergence Divergence).
    
    Args:
        precos (Series): Série de preços
        rapida (int): Período da EMA rápida
        lenta (int): Período da EMA lenta
        sinal (int): Período da linha de sinal
    
    Returns:
        tuple: (macd, macd_signal, macd_histogram)
    """
    ema_rapida = precos.ewm(span=rapida).mean()
    ema_lenta = precos.ewm(span=lenta).mean()
    macd = ema_rapida - ema_lenta
    macd_signal = macd.ewm(span=sinal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calcular_bollinger_bands(precos, periodo=20, std_dev=2):
    """
    Calcula as Bollinger Bands.
    
    Args:
        precos (Series): Série de preços
        periodo (int): Período para média móvel
        std_dev (float): Número de desvios padrão
    
    Returns:
        tuple: (upper, middle, lower)
    """
    sma = precos.rolling(window=periodo).mean()
    std = precos.rolling(window=periodo).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calcular_stochastic(high, low, close, k_periodo=14, d_periodo=3):
    """
    Calcula o Stochastic Oscillator.
    
    Args:
        high (Series): Série de preços máximos
        low (Series): Série de preços mínimos
        close (Series): Série de preços de fechamento
        k_periodo (int): Período para %K
        d_periodo (int): Período para %D
    
    Returns:
        tuple: (k_percent, d_percent)
    """
    lowest_low = low.rolling(window=k_periodo).min()
    highest_high = high.rolling(window=k_periodo).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_periodo).mean()
    return k_percent, d_percent

def calcular_atr(high, low, close, periodo=14):
    """
    Calcula o ATR (Average True Range).
    
    Args:
        high (Series): Série de preços máximos
        low (Series): Série de preços mínimos
        close (Series): Série de preços de fechamento
        periodo (int): Período para cálculo do ATR
    
    Returns:
        Series: Valores do ATR
    """
    high_low = high - low
    high_close_prev = np.abs(high - close.shift())
    low_close_prev = np.abs(low - close.shift())
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr = true_range.rolling(window=periodo).mean()
    return atr

def adicionar_indicadores_tecnicos(df):
    """
    Adiciona indicadores técnicos ao DataFrame.
    
    Args:
        df (DataFrame): DataFrame com dados OHLC
    
    Returns:
        DataFrame: DataFrame com indicadores adicionados
    """
    logger.info("Calculando indicadores técnicos...")
    
    # 1. Retorno Percentual de 1 período
    df['Retorno_Pct'] = df['Close'].pct_change()
    
    # 2. Médias Móveis Simples (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 3. RSI (Relative Strength Index)
    df['RSI'] = calcular_rsi(df['Close'])
    
    # 4. MACD (Moving Average Convergence Divergence)
    macd, macd_signal, macd_hist = calcular_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Histogram'] = macd_hist
    
    # 5. Volume Delta (simplificado)
    # Positivo se candle de alta (Close > Open), negativo se de baixa
    df['Volume_Delta'] = df['Volume'] * (df['Close'] > df['Open']).astype(int) - \
                        df['Volume'] * (df['Close'] <= df['Open']).astype(int)
    
    # 6. Bollinger Bands
    bb_upper, bb_middle, bb_lower = calcular_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb_upper
    df['BB_Middle'] = bb_middle
    df['BB_Lower'] = bb_lower
    
    # 7. Stochastic Oscillator
    stoch_k, stoch_d = calcular_stochastic(df['High'], df['Low'], df['Close'])
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    
    # 8. ATR (Average True Range)
    df['ATR'] = calcular_atr(df['High'], df['Low'], df['Close'])
    
    logger.info("Indicadores técnicos calculados com sucesso")
    return df

def processar_arquivo(caminho_arquivo_entrada, caminho_arquivo_saida):
    """
    Processa um arquivo CSV de dados de mercado, adicionando indicadores técnicos.
    
    Args:
        caminho_arquivo_entrada (str): Caminho para o arquivo CSV de entrada
        caminho_arquivo_saida (str): Caminho para salvar o arquivo processado
    
    Returns:
        bool: True se processado com sucesso, False caso contrário
    """
    try:
        # Leitura do arquivo CSV
        logger.info(f"Lendo arquivo: {os.path.basename(caminho_arquivo_entrada)}")
        
        # Leitura especial para lidar com o formato do yfinance
        with open(caminho_arquivo_entrada, 'r') as f:
            lines = f.readlines()
        
        # Encontrar onde começam os dados reais (pular linha com "Ticker")
        data_start = 0
        for i, line in enumerate(lines):
            if not line.startswith('Price') and not line.startswith('Ticker') and not line.startswith('Date'):
                data_start = i
                break
        
        # Recriar o arquivo temporariamente sem as linhas problemáticas
        clean_lines = ['Price,Close,High,Low,Open,Volume\n'] + lines[data_start:]
        
        # Salvar temporariamente e carregar
        temp_file = caminho_arquivo_entrada + '.temp'
        with open(temp_file, 'w') as f:
            f.writelines(clean_lines)
        
        # Carregar dados limpos
        df = pd.read_csv(temp_file)
        
        # Remover arquivo temporário
        os.remove(temp_file)
        
        # Configurar índice de data corretamente
        df['Date'] = pd.to_datetime(df['Price'])  # A coluna "Price" na verdade contém as datas
        df.set_index('Date', inplace=True)
        df.drop('Price', axis=1, inplace=True)  # Remover coluna Price que era data
        
        # Limpeza: remover linhas com valores nulos
        registros_originais = len(df)
        df.dropna(inplace=True)
        registros_apos_limpeza = len(df)
        
        if registros_apos_limpeza < registros_originais:
            logger.info(f"Removidas {registros_originais - registros_apos_limpeza} linhas com valores nulos")
        
        # Assegurar que o índice é datetime e configurar fuso horário
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert('America/Sao_Paulo')
        
        # Adicionar indicadores técnicos
        df = adicionar_indicadores_tecnicos(df)
        
        # Limpeza final: remover NaNs criados pelos indicadores
        registros_antes_final = len(df)
        df.dropna(inplace=True)
        registros_final = len(df)
        
        if registros_final < registros_antes_final:
            logger.info(f"Removidas {registros_antes_final - registros_final} linhas com NaN dos indicadores")
        
        # Salvar arquivo processado
        df.to_csv(caminho_arquivo_saida)
        
        logger.info(f"Processado: {registros_final} registros finais")
        logger.info(f"Salvo em: {os.path.basename(caminho_arquivo_saida)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro no processamento de {os.path.basename(caminho_arquivo_entrada)}: {str(e)}")
        return False

def processar_dados_brutos():
    """
    Processa todos os arquivos da pasta dados_brutos.
    
    Returns:
        dict: Estatísticas do processamento (sucessos, falhas, total)
    """
    # Caminhos dos diretórios
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dados_brutos_dir = os.path.join(project_root, 'dados_brutos')
    dados_processados_dir = os.path.join(project_root, 'dados_processados')
    
    # Criar diretório de saída se não existir
    if not os.path.exists(dados_processados_dir):
        os.makedirs(dados_processados_dir)
        logger.info(f"Diretório '{dados_processados_dir}' criado!")
    
    # Obter lista de arquivos na pasta dados_brutos
    if not os.path.exists(dados_brutos_dir):
        logger.error(f"Pasta '{dados_brutos_dir}' não encontrada!")
        logger.error("Execute primeiro o script 'coleta_dados.py'")
        return {'sucessos': 0, 'falhas': 0, 'total': 0}
    
    arquivos_brutos = [f for f in os.listdir(dados_brutos_dir) if f.endswith('.csv')]
    
    if not arquivos_brutos:
        logger.error(f"Nenhum arquivo CSV encontrado na pasta '{dados_brutos_dir}'!")
        return {'sucessos': 0, 'falhas': 0, 'total': 0}
    
    logger.info(f"Encontrados {len(arquivos_brutos)} arquivos para processar:")
    for arquivo in sorted(arquivos_brutos):
        logger.info(f"  • {arquivo}")
    
    sucessos = 0
    falhas = 0
    
    # Processar cada arquivo
    for i, arquivo in enumerate(sorted(arquivos_brutos), 1):
        logger.info(f"[{i:2d}/{len(arquivos_brutos)}] Processando {arquivo}...")
        
        # Construir caminhos
        caminho_entrada = os.path.join(dados_brutos_dir, arquivo)
        nome_processado = arquivo.replace('.csv', '_processed.csv')
        caminho_saida = os.path.join(dados_processados_dir, nome_processado)
        
        # Processar arquivo
        if processar_arquivo(caminho_entrada, caminho_saida):
            sucessos += 1
        else:
            falhas += 1
    
    return {
        'sucessos': sucessos,
        'falhas': falhas,
        'total': len(arquivos_brutos)
    }

def main():
    """
    Função principal do script de pré-processamento.
    """
    logger.info("=" * 70)
    logger.info("PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES")
    logger.info("=" * 70)
    
    inicio = datetime.now()
    logger.info(f"Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Processar todos os arquivos
    estatisticas = processar_dados_brutos()
    
    fim = datetime.now()
    duracao = fim - inicio
    
    # Relatório final
    logger.info("=" * 70)
    logger.info("RELATÓRIO FINAL DO PRÉ-PROCESSAMENTO")
    logger.info("=" * 70)
    logger.info(f"✅ Arquivos processados com sucesso: {estatisticas['sucessos']}")
    logger.info(f"❌ Arquivos com falha:               {estatisticas['falhas']}")
    logger.info(f"📊 Total de arquivos:                {estatisticas['total']}")
    logger.info(f"📁 Arquivos salvos em:               ./dados_processados/")
    logger.info(f"⏰ Duração: {duracao}")
    
    if estatisticas['sucessos'] > 0:
        logger.info("🎉 Pré-processamento concluído!")
        logger.info("📈 Indicadores técnicos adicionados:")
        logger.info("   • Retorno Percentual")
        logger.info("   • SMA 20 e 50 períodos")
        logger.info("   • RSI (14 períodos)")
        logger.info("   • MACD com sinal e histograma")
        logger.info("   • Volume Delta")
        logger.info("   • Bollinger Bands")
        logger.info("   • Stochastic Oscillator")
        logger.info("   • ATR (Average True Range)")
        
        # Mostrar exemplo de um arquivo processado
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dados_processados_dir = os.path.join(project_root, 'dados_processados')
        arquivos_processados = [f for f in os.listdir(dados_processados_dir) if f.endswith('.csv')]
        if arquivos_processados:
            logger.info("📄 Exemplo - estrutura do arquivo processado:")
            exemplo_arquivo = os.path.join(dados_processados_dir, arquivos_processados[0])
            df_exemplo = pd.read_csv(exemplo_arquivo, nrows=0)  # Só headers
            logger.info(f"   Colunas: {list(df_exemplo.columns)}")
        
        # Próximo passo
        logger.info("💡 Próximo passo: Execute 'python treinamento_modelo.py'")
    else:
        logger.warning("⚠️ Nenhum arquivo foi processado com sucesso.")
    
    logger.info(f"Fim: {fim.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
