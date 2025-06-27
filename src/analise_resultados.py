"""
ANÁLISE DOS RESULTADOS DO BACKTESTING
=====================================

Este script analisa os resultados do backtesting e propõe melhorias para a estratégia.

Autor: Yan
Data: 2025-06-27
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analise_resultados.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def carregar_resultados():
    """
    Carrega os resultados do backtesting.
    
    Returns:
        DataFrame or None: DataFrame com resultados ou None se não encontrado
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(project_root, 'results', 'resultados_backtest.csv')
    
    if not os.path.exists(results_file):
        logger.error("Arquivo de resultados não encontrado!")
        logger.info("Execute primeiro o script: python backtest_engine.py")
        return None
    
    try:
        df_resultados = pd.read_csv(results_file)
        logger.info(f"Resultados carregados: {len(df_resultados)} ativos")
        return df_resultados
    except Exception as e:
        logger.error(f"Erro ao carregar resultados: {e}")
        return None

def analisar_problemas(df_resultados):
    """
    Analisa os problemas identificados nos resultados.
    
    Args:
        df_resultados (DataFrame): DataFrame com resultados do backtesting
    """
    logger.info("🔍 ANÁLISE DOS PROBLEMAS IDENTIFICADOS:")
    
    # Problema 1: Win Rate vs Retorno
    logger.info("1️⃣ PARADOXO DO WIN RATE:")
    win_rate_medio = df_resultados['win_rate'].mean() * 100
    retorno_medio = df_resultados['retorno_total'].mean() * 100
    
    logger.info(f"   • Win Rate médio: {win_rate_medio:.1f}% (bom)")
    logger.info(f"   • Retorno médio: {retorno_medio:+.2f}% (muito ruim)")
    logger.info("   ➡️ Conclusão: Trades lucrativos pequenos, trades perdedores grandes")
    
    # Problema 2: Stop Loss muito restritivo
    logger.info("2️⃣ GESTÃO DE RISCO:")
    logger.info("   • Stop Loss atual: 5%")
    logger.info("   • Take Profit atual: 10%")
    logger.info("   • Profit Factor alto mas resultado negativo")
    logger.info("   ➡️ Conclusão: Muitos stop losses sendo ativados")
def propor_melhorias():
    """
    Propõe melhorias baseadas na análise dos resultados.
    """
    logger.info("💡 PROPOSTAS DE MELHORIA:")
    
    logger.info("1️⃣ AJUSTAR PARÂMETROS DE RISCO:")
    logger.info("   • Aumentar Stop Loss para 8-10%")
    logger.info("   • Aumentar Take Profit para 15-20%")
    logger.info("   • Melhorar ratio risk/reward")
    
    logger.info("2️⃣ FILTRAR SINAIS:")
    logger.info("   • Aumentar threshold de confiança (0.6 → 0.75)")
    logger.info("   • Adicionar filtros de volatilidade")
    logger.info("   • Considerar contexto de tendência")
    
    logger.info("3️⃣ OTIMIZAR ALGORITMO ZIGZAG:")
    logger.info("   • Testar diferentes thresholds (3% → 5%)")
    logger.info("   • Validar padrões Elliott mais rigorosamente")
    logger.info("   • Adicionar confirmação técnica")
    
    logger.info("4️⃣ MELHORAR FEATURES:")
    logger.info("   • Adicionar indicadores de momento")
    logger.info("   • Incluir análise de volume")
    logger.info("   • Considerar correlação entre ativos")

def criar_backtest_otimizado():
    """
    Cria versão otimizada do backtesting.
    """
    logger.info("=" * 70)
    logger.info("CRIANDO VERSÃO OTIMIZADA DO BACKTESTING")
    logger.info("=" * 70)
    
    # Parâmetros otimizados
    config_otimizada = {
        'capital_inicial': 10000,
        'taxa_corretagem': 0.001,
        'stop_loss': 0.08,           # 8% (era 5%)
        'take_profit': 0.15,         # 15% (era 10%)
        'confianca_minima': 0.75,    # 75% (era 60%)
        'max_posicoes': 2,           # 2 (era 4) - menos posições simultâneas
        'min_intervalo_trades': 5    # Mínimo 5 dias entre trades no mesmo ativo
    }
    
    print("🎯 PARÂMETROS OTIMIZADOS:")
    for key, value in config_otimizada.items():
        if isinstance(value, float) and key in ['stop_loss', 'take_profit']:
            print(f"   • {key}: {value*100:.0f}%")
        elif isinstance(value, float) and key == 'confianca_minima':
            print(f"   • {key}: {value*100:.0f}%")
        else:
            print(f"   • {key}: {value}")
    
    return config_otimizada

def analisar_modelos():
    """Analisa a qualidade dos modelos treinados"""
    print("\n" + "=" * 70)
    print("ANÁLISE DOS MODELOS TREINADOS")
    print("=" * 70)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    if not os.path.exists(models_dir):
        print("Diretório de modelos não encontrado!")
        return
    
    modelos = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    for modelo_file in modelos:
        ativo = modelo_file.replace('modelo_', '').replace('.joblib', '')
        
        # Carregar modelo e métricas
        modelo_data = joblib.load(os.path.join(models_dir, modelo_file))
        metricas = modelo_data.get('metricas', {})
        
        print(f"📊 {ativo.upper()}:")
        print(f"   • Acurácia: {metricas.get('accuracy', 'N/A')}")
        print(f"   • Features: {len(modelo_data.get('feature_names', []))}")
        
        # Analisar importância das features (se disponível)
        modelo_ml = modelo_data['modelo']
        if hasattr(modelo_ml, 'feature_importances_'):
            feature_names = modelo_data.get('feature_names', [])
            importancias = modelo_ml.feature_importances_
            
            # Top 3 features mais importantes
            indices_top = np.argsort(importancias)[-3:][::-1]
            print("   • Top 3 Features:")
            for i, idx in enumerate(indices_top):
                if idx < len(feature_names):
                    print(f"     {i+1}. {feature_names[idx]}: {importancias[idx]:.3f}")
        
        print()

def gerar_recomendacoes():
    """Gera recomendações específicas para melhoria"""
    print("=" * 70)
    print("RECOMENDAÇÕES PARA PRÓXIMAS ITERAÇÕES")
    print("=" * 70)
    
    recomendacoes = [
        {
            'titulo': '🎯 MELHORIA IMEDIATA',
            'itens': [
                'Executar backtest com parâmetros otimizados',
                'Implementar filtro de volatilidade',
                'Adicionar período de "aquecimento" do modelo'
            ]
        },
        {
            'titulo': '📈 MELHORIA DO ALGORITMO',
            'itens': [
                'Implementar Ondas de Elliott mais precisas',
                'Adicionar detecção de divergências',
                'Incluir análise de suporte/resistência'
            ]
        },
        {
            'titulo': '🤖 MELHORIA DO MODELO ML',
            'itens': [
                'Testar outros algoritmos (XGBoost, LSTM)',
                'Implementar ensemble de modelos',
                'Adicionar validação cruzada temporal'
            ]
        },
        {
            'titulo': '📊 MELHORIA DOS DADOS',
            'itens': [
                'Incluir dados de volume mais detalhados',
                'Adicionar dados fundamentalistas',
                'Considerar dados de sentiment do mercado'
            ]
        }
    ]
    
    for rec in recomendacoes:
        print(f"{rec['titulo']}:")
        for item in rec['itens']:
            print(f"   • {item}")
        print()
    
    print("🎉 PRÓXIMOS PASSOS SUGERIDOS:")
    print("1. Implementar backtesting_otimizado.py com novos parâmetros")
    print("2. Testar em dados out-of-sample (dados mais recentes)")
    print("3. Implementar paper trading em tempo real")
    print("4. Desenvolver dashboard de monitoramento")

def analisar_resultados():
    """
    Função principal para análise completa dos resultados.
    
    Returns:
        bool: True se análise foi executada com sucesso
    """
    # Carregar resultados
    df_resultados = carregar_resultados()
    if df_resultados is None:
        return False
    
    logger.info("📊 RESUMO DOS RESULTADOS:")
    logger.info(f"\n{df_resultados.to_string(index=False)}")
    
    # Analisar problemas
    analisar_problemas(df_resultados)
    
    # Analisar frequência de trades
    logger.info("3️⃣ FREQUÊNCIA DE TRADES:")
    total_trades = df_resultados['total_trades'].sum()
    logger.info(f"   • Total de trades: {total_trades}")
    logger.info(f"   • Média por ativo: {total_trades/len(df_resultados):.1f} trades")
    logger.info("   ➡️ Conclusão: Possível overtrading (muitos sinais)")
    
    # Propor melhorias
    propor_melhorias()
    
    return True

def main():
    """
    Função principal do script de análise.
    """
    logger.info("=" * 70)
    logger.info("ANÁLISE DETALHADA DOS RESULTADOS")
    logger.info("=" * 70)
    
    inicio = datetime.now()
    logger.info(f"Início da análise: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Executar análise
    sucesso = analisar_resultados()
    
    if sucesso:
        # Executar análises adicionais
        criar_backtest_otimizado()
        
        # Gerar configuração otimizada
        logger.info("📝 CONFIGURAÇÃO OTIMIZADA RECOMENDADA:")
        logger.info("   • Stop Loss: 8%")
        logger.info("   • Take Profit: 15%")
        logger.info("   • Threshold de confiança: 75%")
        logger.info("   • ZigZag deviation: 5%")
        logger.info("   • Máximo de posições: 2")
    
    fim = datetime.now()
    duracao = fim - inicio
    
    logger.info(f"Fim da análise: {fim.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duração: {duracao}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
