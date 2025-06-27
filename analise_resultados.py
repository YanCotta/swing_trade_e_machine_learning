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
from datetime import datetime

def analisar_resultados():
    """Analisa os resultados do backtesting"""
    print("=" * 70)
    print("ANÁLISE DETALHADA DOS RESULTADOS")
    print("=" * 70)
    
    # Carregar resultados
    if not os.path.exists('resultados_backtest.csv'):
        print("❌ Arquivo de resultados não encontrado!")
        return
    
    df_resultados = pd.read_csv('resultados_backtest.csv')
    
    print("📊 RESUMO DOS RESULTADOS:")
    print(df_resultados.to_string(index=False))
    print()
    
    print("🔍 ANÁLISE DOS PROBLEMAS IDENTIFICADOS:")
    print()
    
    # Problema 1: Win Rate vs Retorno
    print("1️⃣ PARADOXO DO WIN RATE:")
    print("   • Win Rate médio: 56.1% (bom)")
    print("   • Retorno médio: -97.95% (muito ruim)")
    print("   ➡️ Conclusão: Trades lucrativos pequenos, trades perdedores grandes")
    print()
    
    # Problema 2: Stop Loss muito restritivo
    print("2️⃣ GESTÃO DE RISCO:")
    print("   • Stop Loss atual: 5%")
    print("   • Take Profit atual: 10%")
    print("   • Profit Factor alto mas resultado negativo")
    print("   ➡️ Conclusão: Muitos stop losses sendo ativados")
    print()
    
    # Problema 3: Overtrading
    print("3️⃣ FREQUÊNCIA DE TRADES:")
    total_trades = df_resultados['total_trades'].sum()
    print(f"   • Total de trades: {total_trades}")
    print(f"   • Média por ativo: {total_trades/4:.1f} trades")
    print("   ➡️ Conclusão: Possível overtrading (muitos sinais)")
    print()
    
    print("💡 PROPOSTAS DE MELHORIA:")
    print()
    print("1️⃣ AJUSTAR PARÂMETROS DE RISCO:")
    print("   • Aumentar Stop Loss para 8-10%")
    print("   • Aumentar Take Profit para 15-20%")
    print("   • Melhorar ratio risk/reward")
    print()
    print("2️⃣ FILTRAR SINAIS:")
    print("   • Aumentar threshold de confiança (0.6 → 0.75)")
    print("   • Adicionar filtros de volatilidade")
    print("   • Considerar contexto de tendência")
    print()
    print("3️⃣ OTIMIZAR ALGORITMO ZIGZAG:")
    print("   • Testar diferentes thresholds (3% → 5%)")
    print("   • Validar padrões Elliott mais rigorosamente")
    print("   • Adicionar confirmação técnica")
    print()
    print("4️⃣ MELHORAR FEATURES:")
    print("   • Adicionar indicadores de momento")
    print("   • Incluir análise de volume")
    print("   • Considerar correlação entre ativos")

def criar_backtest_otimizado():
    """Cria versão otimizada do backtesting"""
    print("\n" + "=" * 70)
    print("CRIANDO VERSÃO OTIMIZADA DO BACKTESTING")
    print("=" * 70)
    
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
    
    modelos = [f for f in os.listdir('.') if f.startswith('modelo_') and f.endswith('.joblib')]
    
    for modelo_file in modelos:
        ativo = modelo_file.replace('modelo_', '').replace('.joblib', '')
        
        # Carregar modelo e métricas
        modelo_data = joblib.load(modelo_file)
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

def main():
    """Função principal da análise"""
    print(f"Início da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Executar análises
    analisar_resultados()
    config_otimizada = criar_backtest_otimizado()
    analisar_modelos()
    gerar_recomendacoes()
    
    print(f"\nFim da análise: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
