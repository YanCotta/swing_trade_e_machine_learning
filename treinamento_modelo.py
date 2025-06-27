#!/usr/bin/env python3
"""
Script de Rotulagem de Padrões e Treinamento do Modelo
====================================================

Este script implementa a lógica central de detecção de padrões usando ZigZag como proxy
para identificar ondas de Elliott e treina um modelo de Random Forest para classificação.

Abordagem:
- Usa indicador ZigZag para identificar topos e fundos significativos
- Rotula segmentos entre topos/fundos como padrões de impulso ou correção
- Treina um classificador Random Forest para prever futuros padrões
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
from datetime import datetime

def calcular_zigzag(high, low, close, deviation=3.0):
    """
    Implementação manual do indicador ZigZag.
    
    Args:
        high, low, close: Series do pandas com preços
        deviation: Desvio percentual mínimo para considerar um ponto de virada
    
    Returns:
        Series com pontos de virada (NaN para pontos que não são viradas)
    """
    zigzag = pd.Series(index=close.index, dtype=float)
    zigzag.iloc[:] = np.nan
    
    # Inicializar com o primeiro ponto
    trend = None  # 1 para alta, -1 para baixa
    last_pivot_idx = 0
    last_pivot_price = close.iloc[0]
    
    for i in range(1, len(close)):
        current_high = high.iloc[i]
        current_low = low.iloc[i]
        
        # Calcular variação percentual
        high_change = ((current_high - last_pivot_price) / last_pivot_price) * 100
        low_change = ((current_low - last_pivot_price) / last_pivot_price) * 100
        
        if trend is None:
            # Estabelecer primeira tendência
            if high_change >= deviation:
                trend = 1
                zigzag.iloc[last_pivot_idx] = last_pivot_price
                last_pivot_idx = i
                last_pivot_price = current_high
            elif low_change <= -deviation:
                trend = -1
                zigzag.iloc[last_pivot_idx] = last_pivot_price
                last_pivot_idx = i
                last_pivot_price = current_low
        
        elif trend == 1:  # Tendência de alta
            if current_high > last_pivot_price:
                # Novo máximo, atualizar ponto
                last_pivot_idx = i
                last_pivot_price = current_high
            elif low_change <= -deviation:
                # Mudança para baixa
                zigzag.iloc[last_pivot_idx] = last_pivot_price
                trend = -1
                last_pivot_idx = i
                last_pivot_price = current_low
        
        elif trend == -1:  # Tendência de baixa
            if current_low < last_pivot_price:
                # Novo mínimo, atualizar ponto
                last_pivot_idx = i
                last_pivot_price = current_low
            elif high_change >= deviation:
                # Mudança para alta
                zigzag.iloc[last_pivot_idx] = last_pivot_price
                trend = 1
                last_pivot_idx = i
                last_pivot_price = current_high
    
    # Marcar último ponto
    zigzag.iloc[last_pivot_idx] = last_pivot_price
    
    return zigzag

def rotular_dados_zigzag(df, deviation=3.0):
    """
    Rotula os dados usando ZigZag como proxy para ondas de Elliott.
    
    Args:
        df: DataFrame com dados OHLC
        deviation: Desvio percentual para ZigZag
    
    Returns:
        DataFrame com coluna 'Label' adicionada
    """
    print(f"   🔍 Calculando ZigZag com desvio de {deviation}%...")
    
    # Calcular ZigZag
    zigzag = calcular_zigzag(df['High'], df['Low'], df['Close'], deviation)
    df['ZigZag'] = zigzag
    
    # Encontrar pontos de virada válidos
    pontos_virada = df[~df['ZigZag'].isna()].copy()
    
    if len(pontos_virada) < 3:
        print(f"   ⚠️  Poucos pontos de virada encontrados ({len(pontos_virada)})")
        df['Label'] = 0  # Todos indefinidos
        return df
    
    print(f"   📊 Encontrados {len(pontos_virada)} pontos de virada")
    
    # Inicializar labels
    df['Label'] = 0  # 0 = Indefinido/Correção
    
    # Rotular segmentos entre pontos de virada
    for i in range(len(pontos_virada) - 1):
        inicio_idx = pontos_virada.index[i]
        fim_idx = pontos_virada.index[i + 1]
        
        preco_inicio = pontos_virada['ZigZag'].iloc[i]
        preco_fim = pontos_virada['ZigZag'].iloc[i + 1]
        
        # Determinar direção do movimento
        if preco_fim > preco_inicio:
            # Movimento de alta - Impulso de Alta
            label = 1
        else:
            # Movimento de baixa - Impulso de Baixa
            label = 2
        
        # Aplicar label ao segmento
        mask = (df.index >= inicio_idx) & (df.index <= fim_idx)
        df.loc[mask, 'Label'] = label
    
    # Estatísticas dos labels
    label_counts = df['Label'].value_counts()
    print(f"   📈 Labels distribuídos:")
    print(f"      • Indefinido/Correção (0): {label_counts.get(0, 0)}")
    print(f"      • Impulso de Alta (1): {label_counts.get(1, 0)}")
    print(f"      • Impulso de Baixa (2): {label_counts.get(2, 0)}")
    
    return df

def preparar_features_target(df, shift_periods=5):
    """
    Prepara features (X) e target (y) para o modelo.
    
    Args:
        df: DataFrame com dados processados
        shift_periods: Número de períodos para predição futura
    
    Returns:
        X, y: Arrays numpy para treinamento
    """
    print(f"   🔧 Preparando features e target (shift={shift_periods})...")
    
    # Definir colunas de features (indicadores técnicos)
    feature_columns = [
        'Retorno_Pct', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 
        'MACD_Histogram', 'Volume_Delta', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'Stoch_K', 'Stoch_D', 'ATR'
    ]
    
    # Verificar se todas as features existem
    features_disponiveis = [col for col in feature_columns if col in df.columns]
    features_faltando = [col for col in feature_columns if col not in df.columns]
    
    if features_faltando:
        print(f"   ⚠️  Features não encontradas: {features_faltando}")
    
    print(f"   📊 Usando {len(features_disponiveis)} features: {features_disponiveis}")
    
    # Criar features (X)
    X = df[features_disponiveis].copy()
    
    # Criar target (y) - Label deslocada para o futuro
    y = df['Label'].shift(-shift_periods)
    
    # Remover linhas com NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"   📏 Dataset final: {len(X)} amostras, {X.shape[1]} features")
    
    return X.values, y.values, features_disponiveis

def treinar_modelo(X, y, random_state=42):
    """
    Treina o modelo Random Forest.
    
    Args:
        X, y: Dados de treino
        random_state: Seed para reprodutibilidade
    
    Returns:
        modelo treinado, métricas de teste
    """
    print(f"   🤖 Treinando Random Forest...")
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, shuffle=False
    )
    
    print(f"   📊 Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    
    # Treinar modelo
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        max_depth=10,
        min_samples_split=10
    )
    
    modelo.fit(X_train, y_train)
    
    # Avaliação
    y_pred = modelo.predict(X_test)
    
    print(f"   📊 Avaliação do modelo:")
    print("   " + "="*50)
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"   Acurácia geral: {report['accuracy']:.3f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Matriz de Confusão:")
    print(f"   {cm}")
    
    # Importância das features
    feature_importance = modelo.feature_importances_
    
    return modelo, {
        'accuracy': report['accuracy'],
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }

def processar_arquivo_modelo(caminho_arquivo):
    """
    Processa um arquivo para treinamento do modelo.
    
    Args:
        caminho_arquivo: Caminho para arquivo CSV processado
    
    Returns:
        success: Boolean indicando sucesso
    """
    try:
        print(f"   📖 Carregando: {os.path.basename(caminho_arquivo)}")
        
        # Carregar dados
        df = pd.read_csv(caminho_arquivo, index_col=0, parse_dates=True)
        
        if len(df) < 100:
            print(f"   ⚠️  Arquivo muito pequeno ({len(df)} registros)")
            return False
        
        # Rotular dados usando ZigZag
        df = rotular_dados_zigzag(df, deviation=3.0)
        
        # Preparar features e target
        X, y, feature_names = preparar_features_target(df, shift_periods=5)
        
        if len(X) < 50:
            print(f"   ⚠️  Poucos dados para treino ({len(X)} amostras)")
            return False
        
        # Treinar modelo
        modelo, metricas = treinar_modelo(X, y)
        
        # Salvar modelo
        nome_base = os.path.basename(caminho_arquivo).replace('_processed.csv', '')
        caminho_modelo = f'modelo_{nome_base}.joblib'
        joblib.dump({
            'modelo': modelo,
            'feature_names': feature_names,
            'metricas': metricas
        }, caminho_modelo)
        
        print(f"   💾 Modelo salvo: {caminho_modelo}")
        print(f"   ✅ Sucesso!")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erro: {str(e)}")
        return False

def main():
    print("=" * 70)
    print("ROTULAGEM DE DADOS E TREINAMENTO DO MODELO")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Verificar se pasta de dados processados existe
    if not os.path.exists('dados_processados'):
        print("❌ Pasta 'dados_processados' não encontrada!")
        print("   Execute primeiro o script 'preprocessamento.py'")
        return
    
    # Obter arquivos processados
    arquivos_processados = [
        f for f in os.listdir('dados_processados') 
        if f.endswith('_processed.csv')
    ]
    
    if not arquivos_processados:
        print("❌ Nenhum arquivo processado encontrado!")
        return
    
    print(f"📊 Encontrados {len(arquivos_processados)} arquivos processados")
    print("🎯 Estratégia: Treinar um modelo por ativo/timeframe")
    print()
    
    # Para demonstração, vamos treinar apenas com alguns arquivos principais
    arquivos_principais = [
        f for f in arquivos_processados 
        if '_1d_processed.csv' in f  # Focar nos dados diários
    ]
    
    if not arquivos_principais:
        # Se não há dados diários, usar todos
        arquivos_principais = arquivos_processados[:4]  # Primeiros 4 arquivos
    
    print(f"🎯 Treinando modelos para {len(arquivos_principais)} arquivos:")
    for arquivo in arquivos_principais:
        print(f"   • {arquivo}")
    print()
    
    sucessos = 0
    falhas = 0
    
    # Processar cada arquivo
    for i, arquivo in enumerate(arquivos_principais, 1):
        print(f"[{i:2d}/{len(arquivos_principais)}] Processando {arquivo}...")
        
        caminho_arquivo = os.path.join('dados_processados', arquivo)
        
        if processar_arquivo_modelo(caminho_arquivo):
            sucessos += 1
        else:
            falhas += 1
        
        print()  # Linha em branco
    
    # Relatório final
    print("=" * 70)
    print("RELATÓRIO FINAL DO TREINAMENTO")
    print("=" * 70)
    print(f"✅ Modelos treinados com sucesso: {sucessos}")
    print(f"❌ Falhas no treinamento:         {falhas}")
    print(f"📊 Total processado:              {len(arquivos_principais)}")
    print()
    
    if sucessos > 0:
        print("🎉 Treinamento concluído!")
        print("🤖 Modelos disponíveis:")
        
        # Listar modelos criados
        modelos = [f for f in os.listdir('.') if f.startswith('modelo_') and f.endswith('.joblib')]
        for modelo in modelos:
            print(f"   • {modelo}")
        
        print()
        print("💡 Próximo passo: Execute o script de backtest para validar os modelos!")
    else:
        print("⚠️  Nenhum modelo foi treinado com sucesso.")
    
    print(f"\nFim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
