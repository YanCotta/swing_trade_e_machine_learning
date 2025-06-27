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
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def processar_arquivo(caminho_arquivo_entrada, caminho_arquivo_saida):
    """
    Processa um arquivo CSV de dados de mercado, adicionando indicadores técnicos.
    
    Args:
        caminho_arquivo_entrada (str): Caminho para o arquivo CSV de entrada
        caminho_arquivo_saida (str): Caminho para salvar o arquivo processado
    """
    try:
        # Leitura do arquivo CSV
        print(f"   📖 Lendo arquivo: {os.path.basename(caminho_arquivo_entrada)}")
        
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
            print(f"   🧹 Removidas {registros_originais - registros_apos_limpeza} linhas com valores nulos")
        
        # Assegurar que o índice é datetime e configurar fuso horário
        df.index = pd.to_datetime(df.index, utc=True)
        df.index = df.index.tz_convert('America/Sao_Paulo')
        
        # ENGENHARIA DE FEATURES usando cálculos manuais
        print(f"   🔧 Calculando indicadores técnicos...")
        
        # 1. Retorno Percentual de 1 período
        df['Retorno_Pct'] = df['Close'].pct_change()
        
        # 2. Médias Móveis Simples (SMA)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # 3. RSI (Relative Strength Index) - implementação manual
        def calcular_rsi(precos, periodo=14):
            delta = precos.diff()
            ganho = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
            perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
            rs = ganho / perda
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI'] = calcular_rsi(df['Close'])
        
        # 4. MACD (Moving Average Convergence Divergence) - implementação manual
        def calcular_macd(precos, rapida=12, lenta=26, sinal=9):
            ema_rapida = precos.ewm(span=rapida).mean()
            ema_lenta = precos.ewm(span=lenta).mean()
            macd = ema_rapida - ema_lenta
            macd_signal = macd.ewm(span=sinal).mean()
            macd_hist = macd - macd_signal
            return macd, macd_signal, macd_hist
        
        macd, macd_signal, macd_hist = calcular_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_hist
        
        # 5. Volume Delta (simplificado)
        # Positivo se candle de alta (Close > Open), negativo se de baixa
        df['Volume_Delta'] = df['Volume'] * (df['Close'] > df['Open']).astype(int) - \
                            df['Volume'] * (df['Close'] <= df['Open']).astype(int)
        
        # 6. Bollinger Bands - implementação manual
        def calcular_bollinger_bands(precos, periodo=20, std_dev=2):
            sma = precos.rolling(window=periodo).mean()
            std = precos.rolling(window=periodo).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        
        bb_upper, bb_middle, bb_lower = calcular_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        
        # 7. Stochastic Oscillator - implementação manual
        def calcular_stochastic(high, low, close, k_periodo=14, d_periodo=3):
            lowest_low = low.rolling(window=k_periodo).min()
            highest_high = high.rolling(window=k_periodo).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_periodo).mean()
            return k_percent, d_percent
        
        stoch_k, stoch_d = calcular_stochastic(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d
        
        # 8. ATR (Average True Range) - implementação manual
        def calcular_atr(high, low, close, periodo=14):
            high_low = high - low
            high_close_prev = np.abs(high - close.shift())
            low_close_prev = np.abs(low - close.shift())
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            atr = true_range.rolling(window=periodo).mean()
            return atr
        
        df['ATR'] = calcular_atr(df['High'], df['Low'], df['Close'])
        
        # Limpeza final: remover NaNs criados pelos indicadores
        registros_antes_final = len(df)
        df.dropna(inplace=True)
        registros_final = len(df)
        
        if registros_final < registros_antes_final:
            print(f"   🧹 Removidas {registros_antes_final - registros_final} linhas com NaN dos indicadores")
        
        # Salvar arquivo processado
        df.to_csv(caminho_arquivo_saida)
        
        print(f"   ✅ Processado: {registros_final} registros finais")
        print(f"   💾 Salvo em: {os.path.basename(caminho_arquivo_saida)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro no processamento: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Criar diretório de saída se não existir
    if not os.path.exists('dados_processados'):
        os.makedirs('dados_processados')
        print("📁 Diretório 'dados_processados' criado!")
    
    # Obter lista de arquivos na pasta dados_brutos
    if not os.path.exists('dados_brutos'):
        print("❌ Erro: Pasta 'dados_brutos' não encontrada!")
        print("   Execute primeiro o script 'coleta_dados.py'")
        return
    
    arquivos_brutos = [f for f in os.listdir('dados_brutos') if f.endswith('.csv')]
    
    if not arquivos_brutos:
        print("❌ Nenhum arquivo CSV encontrado na pasta 'dados_brutos'!")
        return
    
    print(f"📊 Encontrados {len(arquivos_brutos)} arquivos para processar:")
    for arquivo in sorted(arquivos_brutos):
        print(f"   • {arquivo}")
    print()
    
    sucessos = 0
    falhas = 0
    
    # Processar cada arquivo
    for i, arquivo in enumerate(sorted(arquivos_brutos), 1):
        print(f"[{i:2d}/{len(arquivos_brutos)}] Processando {arquivo}...")
        
        # Construir caminhos
        caminho_entrada = os.path.join('dados_brutos', arquivo)
        nome_processado = arquivo.replace('.csv', '_processed.csv')
        caminho_saida = os.path.join('dados_processados', nome_processado)
        
        # Processar arquivo
        if processar_arquivo(caminho_entrada, caminho_saida):
            sucessos += 1
        else:
            falhas += 1
        
        print()  # Linha em branco entre arquivos
    
    # Relatório final
    print("=" * 70)
    print("RELATÓRIO FINAL DO PRÉ-PROCESSAMENTO")
    print("=" * 70)
    print(f"✅ Arquivos processados com sucesso: {sucessos}")
    print(f"❌ Arquivos com falha:               {falhas}")
    print(f"📊 Total de arquivos:                {len(arquivos_brutos)}")
    print(f"📁 Arquivos salvos em:               ./dados_processados/")
    print()
    
    if sucessos > 0:
        print("🎉 Pré-processamento concluído!")
        print("📈 Indicadores técnicos adicionados:")
        print("   • Retorno Percentual")
        print("   • SMA 20 e 50 períodos")
        print("   • RSI (14 períodos)")
        print("   • MACD com sinal e histograma")
        print("   • Volume Delta")
        print("   • Bollinger Bands")
        print("   • Stochastic Oscillator")
        print("   • ATR (Average True Range)")
        
        # Mostrar exemplo de um arquivo processado
        arquivos_processados = [f for f in os.listdir('dados_processados') if f.endswith('.csv')]
        if arquivos_processados:
            print(f"\n📄 Exemplo - estrutura do arquivo processado:")
            exemplo_arquivo = os.path.join('dados_processados', arquivos_processados[0])
            df_exemplo = pd.read_csv(exemplo_arquivo, nrows=0)  # Só headers
            print(f"   Colunas: {list(df_exemplo.columns)}")
    else:
        print("⚠️  Nenhum arquivo foi processado com sucesso.")
    
    print(f"\nFim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
