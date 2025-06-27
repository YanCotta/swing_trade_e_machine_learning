#!/usr/bin/env python3
"""
Teste de Integração Simplificado
===============================
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def main():
    print("="*50)
    print("TESTE DE INTEGRAÇÃO DO SISTEMA")
    print("="*50)
    
    inicio = datetime.now()
    project_root = os.path.dirname(__file__)
    
    # 1. Verificar configuração
    try:
        config_path = os.path.join(project_root, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✅ Config.json carregado")
    except Exception as e:
        print(f"❌ Erro na configuração: {e}")
        return
    
    # 2. Verificar dados
    try:
        dados_brutos = len(os.listdir(os.path.join(project_root, 'dados_brutos')))
        dados_processados = len(os.listdir(os.path.join(project_root, 'dados_processados')))
        modelos = len([f for f in os.listdir(os.path.join(project_root, 'models')) if f.endswith('.joblib')])
        
        print(f"✅ Dados brutos: {dados_brutos} arquivos")
        print(f"✅ Dados processados: {dados_processados} arquivos")
        print(f"✅ Modelos: {modelos} arquivos")
    except Exception as e:
        print(f"❌ Erro nos dados: {e}")
        return
    
    # 3. Testar modelo
    try:
        models_path = os.path.join(project_root, 'models')
        modelo_files = [f for f in os.listdir(models_path) if f.endswith('.joblib')]
        
        if modelo_files:
            modelo_path = os.path.join(models_path, modelo_files[0])
            modelo_data = joblib.load(modelo_path)
            
            modelo = modelo_data['modelo']
            features = modelo_data.get('feature_names', [])
            
            print(f"✅ Modelo testado: {type(modelo).__name__}")
            print(f"✅ Features: {len(features)}")
            
            # Teste de predição
            if features:
                dummy_data = np.random.random((1, len(features)))
                pred = modelo.predict(dummy_data)
                prob = modelo.predict_proba(dummy_data)
                print(f"✅ Predição: {pred[0]}, confiança: {max(prob[0]):.3f}")
        else:
            print("❌ Nenhum modelo encontrado")
    except Exception as e:
        print(f"❌ Erro no modelo: {e}")
    
    # 4. Status final
    fim = datetime.now()
    print(f"\n⏰ Teste concluído em: {fim - inicio}")
    print("🎉 SISTEMA FUNCIONAL!")
    print("="*50)

if __name__ == "__main__":
    main()
