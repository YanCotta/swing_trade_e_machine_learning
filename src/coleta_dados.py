"""
Script para coleta massiva de dados de mercado da B3 usando yfinance.
Coleta dados históricos em múltiplos timeframes para análise posterior.
"""

import yfinance as yf
import pandas as pd
import os
import json
import logging
from datetime import datetime

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coleta_dados.log'),
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
        # Buscar config.json na raiz do projeto
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
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

def criar_diretorio_dados():
    """
    Cria o diretório de dados brutos se não existir.
    
    Returns:
        bool: True se o diretório existe ou foi criado com sucesso
    """
    dados_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dados_brutos')
    if not os.path.exists(dados_dir):
        try:
            os.makedirs(dados_dir)
            logger.info(f"Diretório '{dados_dir}' criado!")
            return True
        except Exception as e:
            logger.error(f"Erro ao criar diretório '{dados_dir}': {e}")
            return False
    return True

def baixar_dados_ativo(ativo, intervalo, periodo):
    """
    Baixa dados históricos para um ativo específico.
    
    Args:
        ativo (str): Ticker do ativo (ex: 'PETR4.SA')
        intervalo (str): Intervalo dos dados (ex: '1d', '1h')
        periodo (str): Período histórico (ex: '5y', '60d')
    
    Returns:
        tuple: (DataFrame com dados, mensagem de status)
    """
    try:
        logger.info(f"Baixando {ativo} ({intervalo}, {periodo})")
        
        # Download dos dados
        dados = yf.download(
            tickers=ativo,
            interval=intervalo,
            period=periodo,
            progress=False  # Desabilita a barra de progresso para log mais limpo
        )
        
        # Validação dos dados
        if dados.empty:
            logger.warning(f"Dados vazios para {ativo} ({intervalo})")
            return None, "Dados vazios"
        
        logger.info(f"Sucesso: {ativo} ({intervalo}) - {len(dados)} registros")
        return dados, "Sucesso"
        
    except Exception as e:
        logger.error(f"Erro ao baixar {ativo} ({intervalo}): {str(e)}")
        return None, f"Erro: {str(e)}"

def salvar_dados_arquivo(dados, ativo, intervalo):
    """
    Salva os dados em arquivo CSV.
    
    Args:
        dados (DataFrame): Dados a serem salvos
        ativo (str): Ticker do ativo
        intervalo (str): Intervalo dos dados
    
    Returns:
        tuple: (bool sucesso, str caminho_arquivo)
    """
    try:
        # Criar nome do arquivo padronizado
        nome_ativo = ativo.replace('.SA', '')  # Remove sufixo para nome mais limpo
        dados_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dados_brutos')
        nome_arquivo = os.path.join(dados_dir, f"{nome_ativo}_{intervalo}.csv")
        
        # Salvar arquivo
        dados.to_csv(nome_arquivo)
        
        logger.info(f"Arquivo salvo: {nome_arquivo}")
        return True, nome_arquivo
        
    except Exception as e:
        logger.error(f"Erro ao salvar arquivo para {ativo} ({intervalo}): {e}")
        return False, ""

def coletar_dados(config):
    """
    Função principal de coleta de dados para todos os ativos e timeframes.
    
    Args:
        config (dict): Dicionário com configurações do projeto
    
    Returns:
        dict: Estatísticas da coleta (sucessos, falhas, total_arquivos)
    """
    ativos = config['ativos']
    timeframes = config['timeframes']
    
    logger.info(f"Iniciando coleta de dados para {len(ativos)} ativos em {len(timeframes)} timeframes")
    logger.info(f"Ativos: {ativos}")
    logger.info(f"Timeframes: {timeframes}")
    
    # Criar diretório se necessário
    if not criar_diretorio_dados():
        logger.error("Falha ao criar/verificar diretório de dados")
        return {'sucessos': 0, 'falhas': 0, 'total_arquivos': 0}
    
    total_arquivos = 0
    sucessos = 0
    falhas = 0
    
    # Loop principal de coleta
    for ativo in ativos:
        logger.info(f"Processando ativo: {ativo}")
        
        for intervalo, periodo in timeframes.items():
            try:
                # Baixar dados
                dados, status = baixar_dados_ativo(ativo, intervalo, periodo)
                
                if dados is not None:
                    # Salvar arquivo
                    sucesso_salvar, caminho_arquivo = salvar_dados_arquivo(dados, ativo, intervalo)
                    
                    if sucesso_salvar:
                        sucessos += 1
                        total_arquivos += 1
                        logger.info(f"✅ {ativo} ({intervalo}) processado com sucesso")
                    else:
                        falhas += 1
                        logger.error(f"❌ Falha ao salvar {ativo} ({intervalo})")
                else:
                    falhas += 1
                    logger.warning(f"⚠️ Pulando {ativo} ({intervalo}): {status}")
                    
            except Exception as e:
                falhas += 1
                logger.error(f"❌ Erro inesperado ao processar {ativo} ({intervalo}): {e}")
                continue
    
    return {
        'sucessos': sucessos,
        'falhas': falhas,
        'total_arquivos': total_arquivos
    }

def main():
    """
    Função principal do script de coleta de dados.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO COLETA DE DADOS DE MERCADO")
    logger.info("=" * 60)
    
    # Carregar configurações
    config = carregar_configuracao()
    if config is None:
        logger.error("Falha ao carregar configurações. Encerrando.")
        return
    
    # Executar coleta
    inicio = datetime.now()
    logger.info(f"Hora de início: {inicio.strftime('%H:%M:%S')}")
    
    estatisticas = coletar_dados(config)
    
    fim = datetime.now()
    duracao = fim - inicio
    
    # Relatório final
    logger.info("=" * 60)
    logger.info("RELATÓRIO FINAL DA COLETA")
    logger.info("=" * 60)
    logger.info(f"✅ Sucessos: {estatisticas['sucessos']}")
    logger.info(f"❌ Falhas: {estatisticas['falhas']}")
    logger.info(f"📁 Total de arquivos salvos: {estatisticas['total_arquivos']}")
    logger.info(f"⏰ Duração: {duracao}")
    logger.info(f"⏰ Concluído em: {fim.strftime('%H:%M:%S')}")
    
    if estatisticas['sucessos'] > 0:
        logger.info("🎉 Coleta de dados concluída com sucesso!")
        logger.info(f"📂 Arquivos salvos na pasta: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dados_brutos')}")
        
        # Próximo passo
        logger.info("💡 Próximo passo: Execute 'python preprocessamento.py'")
    else:
        logger.warning("⚠️ Nenhum arquivo foi salvo. Verifique sua conexão com a internet.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
