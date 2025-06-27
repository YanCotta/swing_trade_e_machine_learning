#!/usr/bin/env python3
"""
Script de Teste de Integração Completo
=====================================

Este script testa se o pipeline completo está funcionando após os refinamentos.

Autor: Yan
Data: 2025-01-27
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_project_root():
    """Retorna o diretório raiz do projeto"""
    return os.path.dirname(os.path.dirname(__file__))

def carregar_configuracao():
    """Carrega configurações do config.json"""
    try:
        config_path = os.path.join(get_project_root(), 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info("✅ Config.json carregado com sucesso")
        return config
    except Exception as e:
        logger.error(f"❌ Erro ao carregar config.json: {e}")
        return None

def verificar_dados():
    """Verifica se os dados estão disponíveis"""
    resultados = {}
    project_root = get_project_root()
    
    # Dados brutos
    dados_brutos_path = os.path.join(project_root, 'dados_brutos')
    dados_brutos = len([f for f in os.listdir(dados_brutos_path) if f.endswith('.csv')])
    resultados['dados_brutos'] = dados_brutos
    logger.info(f"✅ Dados brutos: {dados_brutos} arquivos")
    
    # Dados processados
    dados_processados_path = os.path.join(project_root, 'dados_processados')
    dados_processados = len([f for f in os.listdir(dados_processados_path) if f.endswith('.csv')])
    resultados['dados_processados'] = dados_processados
    logger.info(f"✅ Dados processados: {dados_processados} arquivos")
    
    # Modelos
    models_path = os.path.join(project_root, 'models')
    modelos = len([f for f in os.listdir(models_path) if f.endswith('.joblib')])
    resultados['modelos'] = modelos
    logger.info(f"✅ Modelos treinados: {modelos} arquivos")
    
    return resultados

def testar_carregamento_modelo():
    """Testa carregamento e estrutura dos modelos"""
    project_root = get_project_root()
    models_path = os.path.join(project_root, 'models')
    modelos = [f for f in os.listdir(models_path) if f.endswith('.joblib')]
    
    if not modelos:
        logger.error("❌ Nenhum modelo encontrado")
        return False
    
    try:
        # Testar primeiro modelo
        modelo_file = modelos[0]
        logger.info(f"🔍 Testando modelo: {modelo_file}")
        
        modelo_path = os.path.join(models_path, modelo_file)
        modelo_data = joblib.load(modelo_path)
        
        # Verificar estrutura esperada
        required_keys = ['modelo', 'feature_names', 'metricas', 'config']
        missing_keys = [key for key in required_keys if key not in modelo_data]
        
        if missing_keys:
            logger.warning(f"⚠️ Chaves faltando no modelo: {missing_keys}")
        else:
            logger.info("✅ Estrutura do modelo OK")
        
        # Testar predição básica
        modelo = modelo_data['modelo']
        feature_names = modelo_data.get('feature_names', [])
        
        logger.info(f"✅ Modelo: {type(modelo).__name__}")
        logger.info(f"✅ Features: {len(feature_names)} variáveis")
        
        # Teste de predição com dados dummy
        if feature_names:
            dummy_data = np.random.random((1, len(feature_names)))
            pred = modelo.predict(dummy_data)
            prob = modelo.predict_proba(dummy_data)
            logger.info(f"✅ Predição teste: {pred[0]}, confiança: {max(prob[0]):.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar modelo: {e}")
        return False

def testar_dados_processados():
    """Testa carregamento dos dados processados"""
    try:
        project_root = get_project_root()
        dados_path = os.path.join(project_root, 'dados_processados')
        arquivos = [f for f in os.listdir(dados_path) if f.endswith('_processed.csv')]
        
        if not arquivos:
            logger.error("❌ Nenhum dado processado encontrado")
            return False
        
        # Testar primeiro arquivo
        arquivo = arquivos[0]
        logger.info(f"🔍 Testando dados: {arquivo}")
        
        arquivo_path = os.path.join(dados_path, arquivo)
        df = pd.read_csv(arquivo_path, index_col=0, parse_dates=True)
        
        logger.info(f"✅ Dados carregados: {len(df)} registros")
        logger.info(f"✅ Colunas: {len(df.columns)} variáveis")
        logger.info(f"✅ Período: {df.index[0]} a {df.index[-1]}")
        
        # Verificar se há features essenciais
        features_essenciais = ['Close', 'Open', 'High', 'Low', 'Volume']
        features_encontradas = [f for f in features_essenciais if f in df.columns]
        logger.info(f"✅ Features OHLCV: {len(features_encontradas)}/5")
        
        # Verificar indicadores técnicos
        indicadores = ['RSI', 'MACD', 'SMA_20', 'SMA_50', 'ATR']
        indicadores_encontrados = [i for i in indicadores if i in df.columns]
        logger.info(f"✅ Indicadores técnicos: {len(indicadores_encontrados)}/{len(indicadores)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao testar dados: {e}")
        return False

def simular_backtest_simples():
    """Simula um backtest básico sem dependências complexas"""
    try:
        logger.info("🔍 Simulando backtest básico...")
        
        project_root = get_project_root()
        models_path = os.path.join(project_root, 'models')
        dados_path = os.path.join(project_root, 'dados_processados')
        
        # Carregar um modelo
        modelos = [f for f in os.listdir(models_path) if f.endswith('.joblib')]
        if not modelos:
            logger.error("❌ Nenhum modelo para teste")
            return False
        
        modelo_file = modelos[0]
        modelo_path = os.path.join(models_path, modelo_file)
        modelo_data = joblib.load(modelo_path)
        modelo = modelo_data['modelo']
        feature_names = modelo_data.get('feature_names', [])
        
        # Carregar dados correspondentes
        nome_modelo = modelo_file.replace('modelo_', '').replace('.joblib', '')
        arquivo_dados = os.path.join(dados_path, f'{nome_modelo}_processed.csv')
        
        if not os.path.exists(arquivo_dados):
            logger.error(f"❌ Dados não encontrados: {arquivo_dados}")
            return False
        
        df = pd.read_csv(arquivo_dados, index_col=0, parse_dates=True)
        
        # Simular algumas predições
        if feature_names and len(feature_names) <= len(df.columns):
            logger.info("✅ Simulando predições...")
            
            # Pegar últimos 10 registros
            df_test = df.tail(10)
            
            # Tentar fazer predições com features disponíveis
            available_features = [f for f in feature_names if f in df_test.columns]
            
            if len(available_features) >= len(feature_names) * 0.8:  # Pelo menos 80% das features
                # Preencher features faltantes com 0
                features_completas = []
                for feature in feature_names:
                    if feature in df_test.columns:
                        features_completas.append(feature)
                    else:
                        df_test[feature] = 0
                        features_completas.append(feature)
                
                X_test = df_test[feature_names].fillna(0).values
                
                predictions = modelo.predict(X_test)
                probabilities = modelo.predict_proba(X_test)
                
                logger.info(f"✅ Predições realizadas: {len(predictions)} sinais")
                logger.info(f"✅ Distribuição: {dict(zip(*np.unique(predictions, return_counts=True)))}")
                logger.info(f"✅ Confiança média: {np.mean([max(p) for p in probabilities]):.3f}")
                
                # Verificar se há sinais de compra/venda
                sinais_compra = sum(predictions == 1)
                sinais_venda = sum(predictions == -1)
                logger.info(f"✅ Sinais: {sinais_compra} compras, {sinais_venda} vendas")
                
                return True
            else:
                logger.warning(f"⚠️ Poucas features compatíveis: {len(available_features)}/{len(feature_names)}")
        
        return False
        
    except Exception as e:
        logger.error(f"❌ Erro na simulação: {e}")
        return False

def testar_importacao_modulos():
    """Testa se os módulos do projeto podem ser importados"""
    try:
        logger.info("🔍 Testando importação dos módulos...")
        
        # Testar importação dos principais módulos
        modulos_testados = 0
        modulos_ok = 0
        
        try:
            import coleta_dados
            logger.info("✅ coleta_dados importado")
            modulos_ok += 1
        except ImportError as e:
            logger.warning(f"⚠️ Erro ao importar coleta_dados: {e}")
        modulos_testados += 1
        
        try:
            import preprocessamento
            logger.info("✅ preprocessamento importado")
            modulos_ok += 1
        except ImportError as e:
            logger.warning(f"⚠️ Erro ao importar preprocessamento: {e}")
        modulos_testados += 1
        
        try:
            import treinamento_modelo
            logger.info("✅ treinamento_modelo importado")
            modulos_ok += 1
        except ImportError as e:
            logger.warning(f"⚠️ Erro ao importar treinamento_modelo: {e}")
        modulos_testados += 1
        
        try:
            import backtest_engine
            logger.info("✅ backtest_engine importado")
            modulos_ok += 1
        except ImportError as e:
            logger.warning(f"⚠️ Erro ao importar backtest_engine: {e}")
        modulos_testados += 1
        
        try:
            import analise_resultados
            logger.info("✅ analise_resultados importado")
            modulos_ok += 1
        except ImportError as e:
            logger.warning(f"⚠️ Erro ao importar analise_resultados: {e}")
        modulos_testados += 1
        
        logger.info(f"✅ Módulos importados: {modulos_ok}/{modulos_testados}")
        return modulos_ok == modulos_testados
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de importação: {e}")
        return False

def executar_pipeline_mini():
    """Executa uma versão mini do pipeline para testar integração"""
    try:
        logger.info("🔍 Executando mini pipeline de integração...")
        
        project_root = get_project_root()
        
        # 1. Verificar se há dados para processar
        dados_path = os.path.join(project_root, 'dados_brutos')
        arquivos_dados = [f for f in os.listdir(dados_path) if f.endswith('.csv')][:2]  # Pegar só 2 arquivos
        
        if not arquivos_dados:
            logger.error("❌ Nenhum dado bruto disponível")
            return False
        
        logger.info(f"✅ Testando com {len(arquivos_dados)} arquivos de dados")
        
        # 2. Verificar modelos
        models_path = os.path.join(project_root, 'models')
        modelos = [f for f in os.listdir(models_path) if f.endswith('.joblib')]
        
        if not modelos:
            logger.error("❌ Nenhum modelo disponível")
            return False
        
        logger.info(f"✅ {len(modelos)} modelos disponíveis")
        
        # 3. Testar carregamento e predição
        for modelo_file in modelos[:2]:  # Testar só 2 modelos
            try:
                modelo_path = os.path.join(models_path, modelo_file)
                modelo_data = joblib.load(modelo_path)
                
                ativo = modelo_file.replace('modelo_', '').replace('.joblib', '')
                logger.info(f"✅ Modelo {ativo} carregado com sucesso")
                
                # Simular predição
                feature_names = modelo_data.get('feature_names', [])
                if feature_names:
                    dummy_data = np.random.random((5, len(feature_names)))
                    modelo = modelo_data['modelo']
                    pred = modelo.predict(dummy_data)
                    logger.info(f"✅ {ativo}: {len(pred)} predições simuladas")
                
            except Exception as e:
                logger.warning(f"⚠️ Erro ao testar modelo {modelo_file}: {e}")
        
        logger.info("✅ Mini pipeline executado com sucesso")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no mini pipeline: {e}")
        return False

def gerar_relatorio_integracao():
    """Gera relatório de integração do sistema"""
    logger.info("=" * 70)
    logger.info("RELATÓRIO DE INTEGRAÇÃO PÓS-REFINAMENTOS")
    logger.info("=" * 70)
    
    inicio = datetime.now()
    
    # Testar componentes
    resultados = {
        'config': carregar_configuracao() is not None,
        'dados': verificar_dados(),
        'modelos': testar_carregamento_modelo(),
        'dados_processados': testar_dados_processados(),
        'importacao_modulos': testar_importacao_modulos(),
        'backtest_simples': simular_backtest_simples(),
        'mini_pipeline': executar_pipeline_mini()
    }
    
    # Resumo
    logger.info("")
    logger.info("📊 RESUMO DOS TESTES:")
    logger.info(f"✅ Configuração: {'OK' if resultados['config'] else 'ERRO'}")
    logger.info(f"✅ Modelos: {'OK' if resultados['modelos'] else 'ERRO'}")
    logger.info(f"✅ Dados Processados: {'OK' if resultados['dados_processados'] else 'ERRO'}")
    logger.info(f"✅ Importação Módulos: {'OK' if resultados['importacao_modulos'] else 'ERRO'}")
    logger.info(f"✅ Simulação Backtest: {'OK' if resultados['backtest_simples'] else 'ERRO'}")
    logger.info(f"✅ Mini Pipeline: {'OK' if resultados['mini_pipeline'] else 'ERRO'}")
    
    # Status geral
    testes_ok = sum([
        resultados['config'],
        resultados['modelos'], 
        resultados['dados_processados'],
        resultados['importacao_modulos'],
        resultados['backtest_simples'],
        resultados['mini_pipeline']
    ])
    total_testes = 6
    
    logger.info("")
    logger.info(f"🎯 STATUS GERAL: {testes_ok}/{total_testes} testes passaram ({testes_ok/total_testes*100:.1f}%)")
    
    if testes_ok == total_testes:
        logger.info("🎉 SISTEMA TOTALMENTE INTEGRADO E FUNCIONAL!")
        logger.info("🚀 Pronto para execução completa do pipeline")
    elif testes_ok >= total_testes * 0.8:
        logger.info("✅ Sistema majoritariamente funcional com pequenos ajustes necessários")
    else:
        logger.info("⚠️ Alguns componentes importantes precisam de atenção")
    
    fim = datetime.now()
    logger.info(f"⏰ Teste concluído em: {fim - inicio}")
    logger.info("=" * 70)
    
    return resultados

def main():
    """Função principal do teste de integração"""
    logger.info("🎯 TESTE DE INTEGRAÇÃO DO SISTEMA REFINADO")
    resultados = gerar_relatorio_integracao()
    
    # Sugestões baseadas nos resultados
    if not all(resultados[key] for key in ['config', 'modelos', 'dados_processados']):
        logger.info("")
        logger.info("💡 SUGESTÕES DE MELHORIAS:")
        
        if not resultados['config']:
            logger.info("- Verificar o arquivo config.json")
        if not resultados['modelos']:
            logger.info("- Executar treinamento de modelos")
        if not resultados['dados_processados']:
            logger.info("- Executar preprocessamento dos dados")
    
    return resultados

if __name__ == "__main__":
    main()
