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
import json
import logging
from datetime import datetime

# Imports opcionais
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("matplotlib não está disponível. Gráficos de feature importance não serão gerados.")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('treinamento_modelo.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def carregar_configuracao():
    """
    Carrega configurações do arquivo config.json.
    
    Returns:
        dict: Dicionário com configurações do projeto ou None se houver erro
    """
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("Configurações carregadas com sucesso")
        return config
    except FileNotFoundError:
        logger.error("Arquivo config.json não encontrado!")
        logger.error("Certifique-se de que o arquivo existe na raiz do projeto.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao ler config.json: {e}")
        return None

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

def rotular_dados_zigzag(df, config):
    """
    Rotula os dados usando ZigZag como proxy para ondas de Elliott.
    
    Args:
        df: DataFrame com dados OHLC
        config: Dicionário com configurações do projeto
    
    Returns:
        DataFrame com coluna 'Label' adicionada
    """
    deviation = config['zigzag_deviation']
    logger.info(f"Calculando ZigZag com desvio de {deviation}%")
    
    # Calcular ZigZag
    zigzag = calcular_zigzag(df['High'], df['Low'], df['Close'], deviation)
    df['ZigZag'] = zigzag
    
    # Encontrar pontos de virada válidos
    pontos_virada = df[~df['ZigZag'].isna()].copy()
    
    if len(pontos_virada) < 3:
        logger.warning(f"Poucos pontos de virada encontrados ({len(pontos_virada)})")
        df['Label'] = 0  # Todos indefinidos
        return df
    
    logger.info(f"Encontrados {len(pontos_virada)} pontos de virada")
    
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
    logger.info("Labels distribuídos:")
    logger.info(f"  • Indefinido/Correção (0): {label_counts.get(0, 0)}")
    logger.info(f"  • Impulso de Alta (1): {label_counts.get(1, 0)}")
    logger.info(f"  • Impulso de Baixa (2): {label_counts.get(2, 0)}")
    
    return df

def preparar_features_target(df, config):
    """
    Prepara features (X) e target (y) para o modelo.
    
    Args:
        df: DataFrame com dados processados
        config: Dicionário com configurações do projeto
    
    Returns:
        X, y: Arrays numpy para treinamento
    """
    shift_periods = abs(config['target_shift'])
    logger.info(f"Preparando features e target (shift={shift_periods})")
    
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
        logger.warning(f"Features não encontradas: {features_faltando}")
    
    logger.info(f"Usando {len(features_disponiveis)} features: {features_disponiveis}")
    
    # Criar features (X)
    X = df[features_disponiveis].copy()
    
    # Criar target (y) - Label deslocada para o futuro
    y = df['Label'].shift(-shift_periods)
    
    # Remover linhas com NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    logger.info(f"Dataset final: {len(X)} amostras, {X.shape[1]} features")
    
    return X.values, y.values, features_disponiveis

def treinar_modelo_temporal(X, y, config, feature_names):
    """
    Treina o modelo usando split temporal para evitar data leakage.
    
    Args:
        X, y: Dados de treino
        config: Dicionário com configurações do projeto
        feature_names: Lista com nomes das features
    
    Returns:
        modelo treinado, métricas de teste
    """
    modelo_config = config.get('modelo', {})
    tipo_modelo = modelo_config.get('tipo', 'random_forest')
    parametros = modelo_config.get('parametros', {})
    
    logger.info(f"Treinando {tipo_modelo} com parâmetros: {parametros}")
    
    # Split temporal: 80% para treino, 20% para teste (sem shuffle)
    split_index = int(0.8 * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    logger.info(f"Split temporal: Treino={len(X_train)} | Teste={len(X_test)}")
    
    # Criar modelo com parâmetros da configuração
    modelo = criar_classificador(tipo_modelo, parametros)
    
    # Treinar modelo
    modelo.fit(X_train, y_train)
    
    # Avaliação
    y_pred = modelo.predict(X_test)
    
    logger.info("Avaliação do modelo:")
    logger.info("=" * 50)
    
    # Relatório de classificação
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"Acurácia geral: {report['accuracy']:.3f}")
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Matriz de Confusão:")
    logger.info(f"{cm}")
    
    # Importância das features (se disponível)
    feature_importance = None
    if hasattr(modelo, 'feature_importances_'):
        feature_importance = modelo.feature_importances_
        
        # Log das features mais importantes
        if feature_names:
            importances = list(zip(feature_names, feature_importance))
            importances.sort(key=lambda x: x[1], reverse=True)
            logger.info("Top 5 features mais importantes:")
            for name, importance in importances[:5]:
                logger.info(f"  {name}: {importance:.3f}")
    
    return modelo, {
        'accuracy': report['accuracy'],
        'classification_report': report,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'feature_names': feature_names
    }

def plotar_feature_importance(modelo, feature_names, nome_modelo):
    """
    Gera e salva gráfico de barras da importância das features.
    
    Args:
        modelo: Modelo treinado com atributo feature_importances_
        feature_names: Lista com nomes das features
        nome_modelo: Nome do modelo para o arquivo
    """
    try:
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib não disponível. Pulando geração de gráfico.")
            return
            
        if not hasattr(modelo, 'feature_importances_') or not feature_names:
            logger.warning("Modelo não tem feature_importances_ ou feature_names não fornecidos")
            return
        
        # Preparar dados para o gráfico
        importances = modelo.feature_importances_
        indices = np.argsort(importances)[::-1]  # Ordenar por importância descendente
        
        # Criar figura
        plt.figure(figsize=(12, 8))
        plt.title(f'Importância das Features - {nome_modelo}', fontsize=16, fontweight='bold')
        
        # Criar gráfico de barras
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        bars = plt.bar(range(len(feature_names)), importances[indices], color=colors)
        
        # Configurar eixos
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importância', fontsize=12)
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
        
        # Adicionar valores nas barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Ajustar layout e salvar
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        # Salvar gráfico
        nome_arquivo = f'feature_importance_{nome_modelo}.png'
        plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
        plt.close()  # Fechar figura para liberar memória
        
        logger.info(f"Gráfico de feature importance salvo: {nome_arquivo}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de feature importance: {e}")

def processar_arquivo_modelo(caminho_arquivo, config):
    """
    Processa um arquivo para treinamento do modelo.
    
    Args:
        caminho_arquivo: Caminho para arquivo CSV processado
        config: Dicionário com configurações do projeto
    
    Returns:
        success: Boolean indicando sucesso
    """
    try:
        logger.info(f"Carregando: {os.path.basename(caminho_arquivo)}")
        
        # Carregar dados
        df = pd.read_csv(caminho_arquivo, index_col=0, parse_dates=True)
        
        if len(df) < 100:
            logger.warning(f"Arquivo muito pequeno ({len(df)} registros)")
            return False
        
        # Rotular dados usando ZigZag
        df = rotular_dados_zigzag(df, config)
        
        # Preparar features e target
        X, y, feature_names = preparar_features_target(df, config)
        
        if len(X) < 50:
            logger.warning(f"Poucos dados para treino ({len(X)} amostras)")
            return False
        
        # Treinar modelo com split temporal
        modelo, metricas = treinar_modelo_temporal(X, y, config, feature_names)
        
        # Gerar gráfico de feature importance (Prompt 4)
        nome_base = os.path.basename(caminho_arquivo).replace('_processed.csv', '')
        plotar_feature_importance(modelo, feature_names, nome_base)
        
        # Salvar modelo
        caminho_modelo = f'modelo_{nome_base}.joblib'
        joblib.dump({
            'modelo': modelo,
            'feature_names': feature_names,
            'metricas': metricas,
            'config': config
        }, caminho_modelo)
        
        logger.info(f"Modelo salvo: {caminho_modelo}")
        logger.info("Sucesso!")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro: {str(e)}")
        return False

def treinar_modelos(config):
    """
    Treina modelos para todos os arquivos processados.
    
    Args:
        config (dict): Configurações do projeto
    
    Returns:
        dict: Estatísticas do treinamento (sucessos, falhas, total)
    """
    # Verificar se pasta de dados processados existe
    if not os.path.exists('dados_processados'):
        logger.error("Pasta 'dados_processados' não encontrada!")
        logger.error("Execute primeiro o script 'preprocessamento.py'")
        return {'sucessos': 0, 'falhas': 0, 'total': 0}
    
    # Obter arquivos processados
    arquivos_processados = [
        f for f in os.listdir('dados_processados') 
        if f.endswith('_processed.csv')
    ]
    
    if not arquivos_processados:
        logger.error("Nenhum arquivo processado encontrado!")
        return {'sucessos': 0, 'falhas': 0, 'total': 0}
    
    logger.info(f"Encontrados {len(arquivos_processados)} arquivos processados")
    logger.info("Estratégia: Treinar um modelo por ativo/timeframe")
    
    # Para demonstração, vamos treinar apenas com alguns arquivos principais
    arquivos_principais = [
        f for f in arquivos_processados 
        if '_1d_processed.csv' in f  # Focar nos dados diários
    ]
    
    if not arquivos_principais:
        # Se não há dados diários, usar todos
        arquivos_principais = arquivos_processados[:4]  # Primeiros 4 arquivos
    
    logger.info(f"Treinando modelos para {len(arquivos_principais)} arquivos:")
    for arquivo in arquivos_principais:
        logger.info(f"  • {arquivo}")
    
    sucessos = 0
    falhas = 0
    
    # Processar cada arquivo
    for i, arquivo in enumerate(arquivos_principais, 1):
        logger.info(f"[{i:2d}/{len(arquivos_principais)}] Processando {arquivo}...")
        
        caminho_arquivo = os.path.join('dados_processados', arquivo)
        
        if processar_arquivo_modelo(caminho_arquivo, config):
            sucessos += 1
        else:
            falhas += 1
    
    return {
        'sucessos': sucessos,
        'falhas': falhas,
        'total': len(arquivos_principais)
    }

def carregar_dados(caminho_arquivo):
    """
    Carrega dados de um arquivo CSV processado.
    
    Args:
        caminho_arquivo (str): Caminho para o arquivo CSV
    
    Returns:
        DataFrame or None: DataFrame carregado ou None em caso de erro
    """
    try:
        logger.info(f"Carregando dados de: {os.path.basename(caminho_arquivo)}")
        df = pd.read_csv(caminho_arquivo, index_col=0, parse_dates=True)
        
        if len(df) < 100:
            logger.warning(f"Arquivo muito pequeno ({len(df)} registros)")
            return None
            
        logger.info(f"Dados carregados: {len(df)} registros")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None

def calcular_features(df):
    """
    Calcula e valida as features necessárias para o modelo.
    
    Args:
        df (DataFrame): DataFrame com dados processados
    
    Returns:
        tuple: (features_array, feature_names) ou (None, None) se erro
    """
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
        logger.warning(f"Features não encontradas: {features_faltando}")
    
    logger.info(f"Usando {len(features_disponiveis)} features: {features_disponiveis}")
    
    try:
        # Criar matrix de features
        X = df[features_disponiveis].copy()
        
        # Verificar se há dados suficientes
        if len(X) < 50:
            logger.warning(f"Poucos dados para treino ({len(X)} amostras)")
            return None, None
            
        return X.values, features_disponiveis
    except Exception as e:
        logger.error(f"Erro ao preparar features: {e}")
        return None, None

def criar_classificador(tipo='random_forest', parametros=None):
    """
    Cria um classificador do tipo especificado.
    
    Args:
        tipo (str): Tipo do classificador ('random_forest', 'xgboost', 'lgbm')
        parametros (dict): Parâmetros específicos do modelo
    
    Returns:
        sklearn classifier: Classificador configurado
    """
    if parametros is None:
        parametros = {}
    
    # Parâmetros padrão para cada tipo
    default_params = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'random_state': 42,
            'n_jobs': -1
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        },
        'lgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }
    }
    
    # Mesclar parâmetros padrão com os fornecidos
    final_params = default_params.get(tipo, default_params['random_forest']).copy()
    final_params.update(parametros)
    
    if tipo == 'random_forest':
        return RandomForestClassifier(**final_params)
    elif tipo == 'xgboost':
        if XGBOOST_AVAILABLE:
            from xgboost import XGBClassifier
            return XGBClassifier(**final_params)
        else:
            logger.warning("XGBoost não disponível, usando Random Forest")
            return RandomForestClassifier(**default_params['random_forest'])
    elif tipo == 'lgbm':
        if LIGHTGBM_AVAILABLE:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**final_params)
        else:
            logger.warning("LightGBM não disponível, usando Random Forest")
            return RandomForestClassifier(**default_params['random_forest'])
    else:
        logger.warning(f"Tipo '{tipo}' não reconhecido, usando Random Forest")
        return RandomForestClassifier(**default_params['random_forest'])

def main():
    """
    Função principal do script de treinamento de modelos.
    """
    logger.info("=" * 70)
    logger.info("ROTULAGEM DE DADOS E TREINAMENTO DO MODELO")
    logger.info("=" * 70)
    
    inicio = datetime.now()
    logger.info(f"Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Carregar configurações
    config = carregar_configuracao()
    if config is None:
        logger.error("Falha ao carregar configurações. Encerrando.")
        return
    
    # Treinar modelos
    estatisticas = treinar_modelos(config)
    
    fim = datetime.now()
    duracao = fim - inicio
    
    # Relatório final
    logger.info("=" * 70)
    logger.info("RELATÓRIO FINAL DO TREINAMENTO")
    logger.info("=" * 70)
    logger.info(f"✅ Modelos treinados com sucesso: {estatisticas['sucessos']}")
    logger.info(f"❌ Falhas no treinamento:         {estatisticas['falhas']}")
    logger.info(f"📊 Total processado:              {estatisticas['total']}")
    logger.info(f"⏰ Duração: {duracao}")
    
    if estatisticas['sucessos'] > 0:
        logger.info("🎉 Treinamento concluído!")
        logger.info("🤖 Modelos disponíveis:")
        
        # Listar modelos criados
        modelos = [f for f in os.listdir('.') if f.startswith('modelo_') and f.endswith('.joblib')]
        for modelo in modelos:
            logger.info(f"   • {modelo}")
        
        logger.info("💡 Próximo passo: Execute o script de backtest para validar os modelos!")
    else:
        logger.warning("⚠️ Nenhum modelo foi treinado com sucesso.")
    
    logger.info(f"Fim: {fim.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
