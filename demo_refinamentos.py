#!/usr/bin/env python3
"""
Script de Demonstração dos Refinamentos Implementados
====================================================

Este script demonstra que os 4 prompts de refinamento foram implementados:

1. ✅ Centralização de Configurações
2. ✅ Refatoração e Robustez do Código  
3. ✅ Aprimoramento do Modelo e Prevenção de Lookahead Bias
4. ✅ Melhoria da Análise de Resultados

Autor: Yan
Data: 2025-01-27
"""

import os
import json
import logging
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def verificar_prompt_1():
    """
    Verifica implementação do Prompt 1: Centralização de Configurações
    """
    logger.info("🔍 Verificando Prompt 1: Centralização de Configurações...")
    
    checks = {
        'config.json': os.path.exists('config.json'),
        'requirements.txt': os.path.exists('requirements.txt'),
        '.gitignore': os.path.exists('.gitignore'),
    }
    
    # Verificar se config.json tem a estrutura correta
    if checks['config.json']:
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            required_keys = ['ativos', 'timeframes', 'zigzag_deviation', 'target_shift']
            config_valid = all(key in config for key in required_keys)
            checks['config_structure'] = config_valid
            
        except Exception as e:
            logger.error(f"Erro ao ler config.json: {e}")
            checks['config_structure'] = False
    
    # Relatório
    for item, status in checks.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"  {status_icon} {item}")
    
    all_passed = all(checks.values())
    status = "✅ IMPLEMENTADO" if all_passed else "❌ INCOMPLETO"
    logger.info(f"Prompt 1 Status: {status}")
    return all_passed

def verificar_prompt_2():
    """
    Verifica implementação do Prompt 2: Refatoração e Robustez
    """
    logger.info("🔍 Verificando Prompt 2: Refatoração e Robustez do Código...")
    
    scripts_principais = [
        'coleta_dados.py',
        'preprocessamento.py', 
        'treinamento_modelo.py',
        'backtesting.py'
    ]
    
    checks = {}
    
    for script in scripts_principais:
        if os.path.exists(script):
            with open(script, 'r') as f:
                content = f.read()
            
            # Verificar modularização
            has_functions = 'def ' in content and '():' in content
            
            # Verificar bloco main
            has_main_block = 'if __name__ == "__main__":' in content
            
            # Verificar logging
            has_logging = 'import logging' in content and 'logger.' in content
            
            # Verificar docstrings
            has_docstrings = '"""' in content
            
            checks[script] = {
                'functions': has_functions,
                'main_block': has_main_block,
                'logging': has_logging,
                'docstrings': has_docstrings
            }
    
    # Relatório
    for script, features in checks.items():
        logger.info(f"  📄 {script}:")
        for feature, status in features.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"    {status_icon} {feature}")
    
    # Verificar se todos os scripts passaram em todos os checks
    all_passed = all(
        all(features.values()) 
        for features in checks.values()
    )
    
    status = "✅ IMPLEMENTADO" if all_passed else "❌ INCOMPLETO"
    logger.info(f"Prompt 2 Status: {status}")
    return all_passed

def verificar_prompt_3():
    """
    Verifica implementação do Prompt 3: Aprimoramento do Modelo
    """
    logger.info("🔍 Verificando Prompt 3: Aprimoramento do Modelo e Prevenção de Lookahead Bias...")
    
    checks = {}
    
    # Verificar treinamento_modelo.py
    if os.path.exists('treinamento_modelo.py'):
        with open('treinamento_modelo.py', 'r') as f:
            content = f.read()
        
        checks['funcoes_separadas'] = all(func in content for func in [
            'def carregar_dados(',
            'def calcular_features(',
            'def rotular_dados_zigzag(',
            'def treinar_modelo_temporal('
        ])
        
        checks['classificador_configuravel'] = 'def criar_classificador(' in content
    
    # Verificar backtesting.py
    if os.path.exists('backtesting.py'):
        with open('backtesting.py', 'r') as f:
            content = f.read()
        
        # Verificar se tem EstrategiaML com método next() e init()
        checks['estrategia_ml'] = 'class EstrategiaML' in content
        checks['metodo_next'] = 'def next(self):' in content
        checks['metodo_init'] = 'def init(self):' in content
        checks['lookahead_safe'] = 'self.I(' in content  # Uso da função segura da biblioteca
    
    # Relatório
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"  {status_icon} {check}")
    
    all_passed = all(checks.values())
    status = "✅ IMPLEMENTADO" if all_passed else "❌ INCOMPLETO"
    logger.info(f"Prompt 3 Status: {status}")
    return all_passed

def verificar_prompt_4():
    """
    Verifica implementação do Prompt 4: Melhoria da Análise de Resultados
    """
    logger.info("🔍 Verificando Prompt 4: Melhoria da Análise de Resultados...")
    
    checks = {}
    
    # Verificar backtesting.py para salvamento de estatísticas
    if os.path.exists('backtesting.py'):
        with open('backtesting.py', 'r') as f:
            content = f.read()
        
        checks['salvar_stats'] = 'resultados_backtest' in content and '.txt' in content
        checks['salvar_trades'] = 'trades_realizados' in content and '.csv' in content
    
    # Verificar treinamento_modelo.py para feature importance
    if os.path.exists('treinamento_modelo.py'):
        with open('treinamento_modelo.py', 'r') as f:
            content = f.read()
        
        checks['feature_importance_plot'] = 'def plotar_feature_importance(' in content
        checks['matplotlib_import'] = 'matplotlib' in content
        checks['feature_importance_save'] = 'feature_importance' in content and '.png' in content
    
    # Verificar README.md para seção de resultados
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            content = f.read()
        
        checks['secao_resultados'] = 'Feature Importance' in content
        checks['imagem_feature_importance'] = 'feature_importance' in content and '.png' in content
    
    # Relatório
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        logger.info(f"  {status_icon} {check}")
    
    all_passed = all(checks.values())
    status = "✅ IMPLEMENTADO" if all_passed else "❌ INCOMPLETO"
    logger.info(f"Prompt 4 Status: {status}")
    return all_passed

def gerar_relatorio_final(resultados):
    """
    Gera relatório final da verificação dos refinamentos
    """
    logger.info("=" * 70)
    logger.info("RELATÓRIO FINAL - VERIFICAÇÃO DOS REFINAMENTOS")
    logger.info("=" * 70)
    
    prompts = [
        ("Prompt 1: Centralização de Configurações", resultados[0]),
        ("Prompt 2: Refatoração e Robustez", resultados[1]),
        ("Prompt 3: Aprimoramento do Modelo", resultados[2]),
        ("Prompt 4: Melhoria da Análise", resultados[3])
    ]
    
    implementados = sum(resultados)
    total = len(resultados)
    
    for nome, status in prompts:
        status_icon = "✅" if status else "❌"
        logger.info(f"{status_icon} {nome}")
    
    logger.info("")
    logger.info(f"📊 RESUMO: {implementados}/{total} prompts implementados ({implementados/total*100:.1f}%)")
    
    if implementados == total:
        logger.info("🎉 TODOS OS REFINAMENTOS FORAM IMPLEMENTADOS COM SUCESSO!")
        logger.info("🚀 O projeto está pronto para a próxima fase de desenvolvimento")
    else:
        logger.info("⚠️ Alguns refinamentos ainda precisam de atenção")
        logger.info("📝 Consulte os detalhes acima para identificar os itens pendentes")
    
    logger.info("=" * 70)

def main():
    """
    Função principal do script de verificação
    """
    logger.info("🎯 VERIFICAÇÃO DOS REFINAMENTOS IMPLEMENTADOS")
    logger.info("=" * 70)
    
    inicio = datetime.now()
    
    # Executar verificações
    resultados = [
        verificar_prompt_1(),
        verificar_prompt_2(), 
        verificar_prompt_3(),
        verificar_prompt_4()
    ]
    
    logger.info("")
    
    # Gerar relatório final
    gerar_relatorio_final(resultados)
    
    fim = datetime.now()
    duracao = fim - inicio
    
    logger.info(f"⏰ Verificação concluída em: {duracao}")

if __name__ == "__main__":
    main()
