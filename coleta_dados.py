"""
Script para coleta massiva de dados de mercado da B3 usando yfinance.
Coleta dados históricos em múltiplos timeframes para análise posterior.
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def main():
    # Lista de ativos da B3 (sufixo .SA é obrigatório para o Yahoo Finance)
    ativos = ['PETR4.SA', 'VALE3.SA', 'BBAS3.SA', 'BOVA11.SA']
    
    # Dicionário com timeframes e períodos compatíveis com yfinance
    timeframes = {
        '1d': '5y',    # 5 anos para dados diários
        '4h': '2y',    # 2 anos para dados de 4 horas (yfinance converte para 1h)
        '15m': '60d',  # 60 dias para dados de 15 minutos
        '5m': '60d',   # 60 dias para dados de 5 minutos
        '1m': '7d'     # 7 dias para dados de 1 minuto
    }
    
    # Criar diretório de dados brutos se não existir
    if not os.path.exists('dados_brutos'):
        os.makedirs('dados_brutos')
        print("📁 Diretório 'dados_brutos' criado!")
    
    print(f"🚀 Iniciando coleta de dados para {len(ativos)} ativos em {len(timeframes)} timeframes...")
    print(f"⏰ Hora de início: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    total_arquivos = 0
    sucessos = 0
    falhas = 0
    
    # Loop principal de coleta
    for ativo in ativos:
        print(f"\n📊 Processando ativo: {ativo}")
        
        for intervalo, periodo in timeframes.items():
            try:
                print(f"   ⏬ Baixando {ativo} ({intervalo}, {periodo})...", end=" ")
                
                # Download dos dados
                dados = yf.download(
                    tickers=ativo,
                    interval=intervalo,
                    period=periodo,
                    progress=False  # Desabilita a barra de progresso para log mais limpo
                )
                
                # Validação dos dados
                if dados.empty:
                    print("❌ Dados vazios!")
                    falhas += 1
                    continue
                
                # Criar nome do arquivo padronizado
                nome_ativo = ativo.replace('.SA', '')  # Remove sufixo para nome mais limpo
                nome_arquivo = f"dados_brutos/{nome_ativo}_{intervalo}.csv"
                
                # Salvar arquivo
                dados.to_csv(nome_arquivo)
                
                print(f"✅ Salvo! ({len(dados)} registros)")
                sucessos += 1
                total_arquivos += 1
                
            except Exception as e:
                print(f"❌ Erro: {str(e)}")
                falhas += 1
                continue
    
    # Relatório final
    print("\n" + "=" * 60)
    print("📋 RELATÓRIO FINAL DA COLETA")
    print("=" * 60)
    print(f"✅ Sucessos: {sucessos}")
    print(f"❌ Falhas: {falhas}")
    print(f"📁 Total de arquivos salvos: {total_arquivos}")
    print(f"⏰ Concluído em: {datetime.now().strftime('%H:%M:%S')}")
    
    if sucessos > 0:
        print("\n🎉 Coleta de dados concluída com sucesso!")
        print("📂 Arquivos salvos na pasta 'dados_brutos/'")
    else:
        print("\n⚠️  Nenhum arquivo foi salvo. Verifique sua conexão com a internet.")

if __name__ == "__main__":
    main()
