"""
BACKTESTING DA ESTRATÉGIA DE SWING TRADING
==========================================

Este script implementa o backtest completo da estratégia de trading baseada em Machine Learning
e Teoria das Ondas de Elliott, calculando métricas de performance e risco.

Autor: Yan
Data: 2025-06-27
"""

import os
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import warnings
from backtesting import Backtest, Strategy
import pandas_ta as ta

warnings.filterwarnings('ignore')

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtesting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Engine de backtesting para estratégia de swing trading"""
    
    def __init__(self, capital_inicial=10000, taxa_corretagem=0.001, 
                 stop_loss=0.05, take_profit=0.10, max_posicoes=4):
        """
        Inicializa o engine de backtesting
        
        Args:
            capital_inicial (float): Capital inicial em R$
            taxa_corretagem (float): Taxa de corretagem por operação (0.1% = 0.001)
            stop_loss (float): Stop loss em percentual (5% = 0.05)
            take_profit (float): Take profit em percentual (10% = 0.10)
            max_posicoes (int): Máximo de posições simultâneas
        """
        self.capital_inicial = capital_inicial
        self.taxa_corretagem = taxa_corretagem
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_posicoes = max_posicoes
        
        # Estado do portfólio
        self.capital_atual = capital_inicial
        self.posicoes_abertas = {}
        self.historico_trades = []
        self.historico_capital = []
        
        # Métricas
        self.total_trades = 0
        self.trades_lucrativos = 0
        self.trades_perdedores = 0
        self.lucro_total = 0
        self.perda_total = 0
        
    def executar_backtest(self, dados, modelo, ativo):
        """
        Executa o backtest para um ativo específico
        
        Args:
            dados (DataFrame): Dados históricos processados
            modelo: Modelo treinado
            ativo (str): Nome do ativo
            
        Returns:
            dict: Resultados do backtest
        """
        logger.info(f"Executando backtest para {ativo}...")
        
        # Reset do estado
        self.capital_atual = self.capital_inicial
        self.posicoes_abertas = {}
        self.historico_trades = []
        self.historico_capital = []
        
        # Features para predição
        features = ['Retorno_Pct', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 
                   'MACD_Histogram', 'Volume_Delta', 'BB_Upper', 'BB_Middle', 'BB_Lower', 
                   'Stoch_K', 'Stoch_D', 'ATR']
        
        # Preparar dados
        dados = dados.dropna()
        X = dados[features].values
        dates = dados.index
        precos = dados['Close'].values
        
        # Executar simulação
        for i in range(len(dados)):
            data_atual = dates[i]
            preco_atual = precos[i]
            
            # Fazer predição
            if i >= 5:  # Aguardar dados suficientes
                predicao = modelo.predict(X[i:i+1])[0]
                confianca = max(modelo.predict_proba(X[i:i+1])[0])
                
                # Gerenciar posições abertas
                self._gerenciar_posicoes(data_atual, preco_atual, ativo)
                
                # Estratégia de entrada
                if predicao == 1 and confianca > 0.6:  # Sinal de alta forte
                    self._abrir_posicao_compra(data_atual, preco_atual, ativo)
                elif predicao == 2 and confianca > 0.6:  # Sinal de baixa forte
                    self._abrir_posicao_venda(data_atual, preco_atual, ativo)
            
            # Registrar capital
            self.historico_capital.append({
                'Data': data_atual,
                'Capital': self.capital_atual + self._calcular_valor_posicoes(preco_atual),
                'Ativo': ativo
            })
        
        # Fechar posições restantes
        self._fechar_todas_posicoes(dates[-1], precos[-1], ativo)
        
        return self._calcular_metricas(ativo)
    
    def _abrir_posicao_compra(self, data, preco, ativo):
        """Abre posição de compra (long)"""
        if len(self.posicoes_abertas) >= self.max_posicoes:
            return
        
        # Calcular tamanho da posição (dividir capital por max_posicoes)
        valor_posicao = self.capital_atual / self.max_posicoes
        quantidade = int(valor_posicao / preco)
        
        if quantidade > 0:
            custo_total = quantidade * preco * (1 + self.taxa_corretagem)
            
            if custo_total <= self.capital_atual:
                self.capital_atual -= custo_total
                
                self.posicoes_abertas[f"{ativo}_{len(self.posicoes_abertas)}"] = {
                    'tipo': 'LONG',
                    'ativo': ativo,
                    'data_entrada': data,
                    'preco_entrada': preco,
                    'quantidade': quantidade,
                    'valor_entrada': custo_total,
                    'stop_loss': preco * (1 - self.stop_loss),
                    'take_profit': preco * (1 + self.take_profit)
                }
    
    def _abrir_posicao_venda(self, data, preco, ativo):
        """Abre posição de venda (short) - simplificado"""
        # Para simplicidade, vamos apenas evitar compras em sinais de baixa
        pass
    
    def _gerenciar_posicoes(self, data, preco, ativo):
        """Gerencia posições abertas (stop loss e take profit)"""
        posicoes_para_fechar = []
        
        for key, posicao in self.posicoes_abertas.items():
            if posicao['ativo'] == ativo:
                # Verificar stop loss
                if preco <= posicao['stop_loss']:
                    self._fechar_posicao(key, data, preco, 'STOP_LOSS')
                    posicoes_para_fechar.append(key)
                # Verificar take profit
                elif preco >= posicao['take_profit']:
                    self._fechar_posicao(key, data, preco, 'TAKE_PROFIT')
                    posicoes_para_fechar.append(key)
        
        # Remover posições fechadas
        for key in posicoes_para_fechar:
            del self.posicoes_abertas[key]
    
    def _fechar_posicao(self, key, data, preco, motivo):
        """Fecha uma posição específica"""
        posicao = self.posicoes_abertas[key]
        
        # Calcular resultado
        valor_venda = posicao['quantidade'] * preco * (1 - self.taxa_corretagem)
        resultado = valor_venda - posicao['valor_entrada']
        resultado_pct = resultado / posicao['valor_entrada']
        
        # Atualizar capital
        self.capital_atual += valor_venda
        
        # Registrar trade
        trade = {
            'ativo': posicao['ativo'],
            'data_entrada': posicao['data_entrada'],
            'data_saida': data,
            'preco_entrada': posicao['preco_entrada'],
            'preco_saida': preco,
            'quantidade': posicao['quantidade'],
            'resultado': resultado,
            'resultado_pct': resultado_pct,
            'motivo': motivo
        }
        
        self.historico_trades.append(trade)
        
        # Atualizar estatísticas
        self.total_trades += 1
        if resultado > 0:
            self.trades_lucrativos += 1
            self.lucro_total += resultado
        else:
            self.trades_perdedores += 1
            self.perda_total += abs(resultado)
    
    def _fechar_todas_posicoes(self, data, preco, ativo):
        """Fecha todas as posições abertas"""
        posicoes_para_fechar = list(self.posicoes_abertas.keys())
        for key in posicoes_para_fechar:
            if self.posicoes_abertas[key]['ativo'] == ativo:
                self._fechar_posicao(key, data, preco, 'FIM_BACKTEST')
                del self.posicoes_abertas[key]
    
    def _calcular_valor_posicoes(self, preco_atual):
        """Calcula valor atual das posições abertas"""
        valor_total = 0
        for posicao in self.posicoes_abertas.values():
            valor_total += posicao['quantidade'] * preco_atual
        return valor_total
    
    def _calcular_metricas(self, ativo):
        """Calcula métricas de performance"""
        if not self.historico_trades:
            return {
                'ativo': ativo,
                'capital_final': self.capital_atual,
                'retorno_total': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Calcular métricas básicas
        capital_final = self.capital_atual
        retorno_total = (capital_final - self.capital_inicial) / self.capital_inicial
        win_rate = self.trades_lucrativos / self.total_trades if self.total_trades > 0 else 0
        profit_factor = self.lucro_total / self.perda_total if self.perda_total > 0 else float('inf')
        
        # Calcular drawdown
        valores_capital = [item['Capital'] for item in self.historico_capital]
        peak = self.capital_inicial
        max_drawdown = 0
        
        for valor in valores_capital:
            if valor > peak:
                peak = valor
            drawdown = (peak - valor) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calcular Sharpe ratio simplificado
        if len(self.historico_trades) > 1:
            retornos = [trade['resultado_pct'] for trade in self.historico_trades]
            sharpe_ratio = np.mean(retornos) / np.std(retornos) if np.std(retornos) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'ativo': ativo,
            'capital_final': capital_final,
            'retorno_total': retorno_total,
            'total_trades': self.total_trades,
            'trades_lucrativos': self.trades_lucrativos,
            'trades_perdedores': self.trades_perdedores,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'lucro_medio': self.lucro_total / self.trades_lucrativos if self.trades_lucrativos > 0 else 0,
            'perda_media': self.perda_total / self.trades_perdedores if self.trades_perdedores > 0 else 0
        }

def carregar_modelos_disponiveis():
    """
    Carrega lista de modelos disponíveis para backtesting.
    
    Returns:
        list: Lista de arquivos de modelos disponíveis
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    
    if not os.path.exists(models_dir):
        logger.error("Diretório de modelos não encontrado!")
        logger.info("Execute primeiro o script: python treinamento_modelo.py")
        return []
    
    modelos_disponiveis = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    if not modelos_disponiveis:
        logger.error("Nenhum modelo treinado encontrado!")
        logger.info("Execute primeiro o script: python treinamento_modelo.py")
        return []
    
    logger.info(f"Encontrados {len(modelos_disponiveis)} modelos treinados")
    return [(models_dir, f) for f in modelos_disponiveis]

def executar_backtest_modelo(modelo_info, engine, dados_dir):
    """
    Executa backtest para um modelo específico.
    
    Args:
        modelo_info (tuple): (models_dir, modelo_file)
        engine (BacktestEngine): Engine de backtesting
        dados_dir (str): Diretório dos dados processados
    
    Returns:
        dict or None: Resultado do backtest ou None se houver erro
    """
    try:
        models_dir, modelo_file = modelo_info
        
        # Extrair informações do nome do arquivo
        parts = modelo_file.replace('modelo_', '').replace('.joblib', '').split('_')
        ativo = parts[0]
        timeframe = parts[1]
        
        logger.info(f"Processando {ativo} ({timeframe})...")
        
        # Carregar modelo
        modelo_data = joblib.load(os.path.join(models_dir, modelo_file))
        modelo = modelo_data['modelo']  # Extrair o modelo do dicionário
        
        # Carregar dados
        dados_file = f"{dados_dir}/{ativo}_{timeframe}_processed.csv"
        if not os.path.exists(dados_file):
            logger.warning(f"Arquivo de dados não encontrado: {dados_file}")
            return None
        
        dados = pd.read_csv(dados_file, index_col=0, parse_dates=True)
        
        # Executar backtest
        resultado = engine.executar_backtest(dados, modelo, f"{ativo}_{timeframe}")
        
        logger.info("Backtest concluído!")
        logger.info(f"Retorno: {resultado['retorno_total']*100:+.2f}%")
        logger.info(f"Win Rate: {resultado['win_rate']*100:.1f}%")
        logger.info(f"Total Trades: {resultado['total_trades']}")
        
        return resultado
        
    except Exception as e:
        logger.error(f"Erro no backtest de {modelo_file}: {str(e)}")
        return None

def gerar_relatorio_final(resultados):
    """
    Gera relatório final dos resultados de backtesting.
    
    Args:
        resultados (list): Lista de resultados dos backtests
    """
    if not resultados:
        logger.error("Nenhum backtest foi executado com sucesso!")
        return
    
    logger.info("=" * 70)
    logger.info("RELATÓRIO FINAL DO BACKTESTING")
    logger.info("=" * 70)
    
    # Resumo por ativo
    for resultado in resultados:
        logger.info(f"📊 {resultado['ativo']}:")
        logger.info(f"   💰 Capital Final: R$ {resultado['capital_final']:,.2f}")
        logger.info(f"   📈 Retorno Total: {resultado['retorno_total']*100:+.2f}%")
        logger.info(f"   🎯 Win Rate: {resultado['win_rate']*100:.1f}%")
        logger.info(f"   📊 Total Trades: {resultado['total_trades']}")
        logger.info(f"   🎲 Profit Factor: {resultado['profit_factor']:.2f}")
        logger.info(f"   📉 Max Drawdown: {resultado['max_drawdown']*100:.2f}%")
        logger.info(f"   📈 Sharpe Ratio: {resultado['sharpe_ratio']:.2f}")
    
    # Estatísticas consolidadas
    capital_total = sum(r['capital_final'] for r in resultados)
    retorno_medio = np.mean([r['retorno_total'] for r in resultados])
    win_rate_medio = np.mean([r['win_rate'] for r in resultados])
    total_trades = sum(r['total_trades'] for r in resultados)
    
    logger.info("📊 CONSOLIDADO:")
    logger.info(f"   💰 Capital Total: R$ {capital_total:,.2f}")
    logger.info(f"   📈 Retorno Médio: {retorno_medio*100:+.2f}%")
    logger.info(f"   🎯 Win Rate Médio: {win_rate_medio*100:.1f}%")
    logger.info(f"   📊 Total de Trades: {total_trades}")
    
    # Performance vs Buy & Hold
    logger.info("🎯 ANÁLISE DE PERFORMANCE:")
    melhor_resultado = max(resultados, key=lambda x: x['retorno_total'])
    pior_resultado = min(resultados, key=lambda x: x['retorno_total'])
    
    logger.info(f"   🏆 Melhor Performance: {melhor_resultado['ativo']} ({melhor_resultado['retorno_total']*100:+.2f}%)")
    logger.info(f"   📉 Pior Performance: {pior_resultado['ativo']} ({pior_resultado['retorno_total']*100:+.2f}%)")
    
    # Salvar resultados
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    df_resultados = pd.DataFrame(resultados)
    resultados_file = os.path.join(results_dir, 'resultados_backtest.csv')
    df_resultados.to_csv(resultados_file, index=False)
    logger.info(f"💾 Resultados salvos em: {resultados_file}")

def executar_backtesting():
    """
    Função principal para executar todos os backtests.
    
    Returns:
        list: Lista de resultados dos backtests
    """
    # Configurações
    CAPITAL_INICIAL = 10000
    TAXA_CORRETAGEM = 0.001  # 0.1%
    STOP_LOSS = 0.05         # 5%
    TAKE_PROFIT = 0.10       # 10%
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dados_dir = os.path.join(project_root, 'dados_processados')
    
    # Verificações iniciais
    if not os.path.exists(dados_dir):
        logger.error(f"Diretório '{dados_dir}' não encontrado!")
        logger.info("Execute primeiro os scripts de coleta e pré-processamento")
        return []
    
    # Carregar modelos
    modelos_disponiveis = carregar_modelos_disponiveis()
    if not modelos_disponiveis:
        return []
    
    logger.info(f"💰 Capital inicial: R$ {CAPITAL_INICIAL:,.2f}")
    logger.info("📊 Configurações:")
    logger.info(f"   • Taxa de corretagem: {TAXA_CORRETAGEM*100:.1f}%")
    logger.info(f"   • Stop Loss: {STOP_LOSS*100:.0f}%")
    logger.info(f"   • Take Profit: {TAKE_PROFIT*100:.0f}%")
    
    # Inicializar engine de backtesting
    engine = BacktestEngine(
        capital_inicial=CAPITAL_INICIAL,
        taxa_corretagem=TAXA_CORRETAGEM,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT
    )
    
    # Executar backtests
    resultados = []
    
    for modelo_info in modelos_disponiveis:
        resultado = executar_backtest_modelo(modelo_info, engine, dados_dir)
        if resultado:
            resultados.append(resultado)
    
    return resultados

def executar_backtesting_completo():
    """
    Função principal para executar todos os backtests usando a biblioteca backtesting.
    
    Returns:
        list: Lista de resultados dos backtests
    """
    # Configurações
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dados_dir = os.path.join(project_root, 'dados_processados')
    
    # Verificações iniciais
    if not os.path.exists(dados_dir):
        logger.error(f"Diretório '{dados_dir}' não encontrado!")
        logger.info("Execute primeiro os scripts de coleta e pré-processamento")
        return []
    
    # Carregar modelos
    modelos_disponiveis = carregar_modelos_disponiveis()
    if not modelos_disponiveis:
        return []
    
    logger.info("📊 Executando backtests com biblioteca backtesting (sem lookahead bias)")
    
    # Executar backtests
    resultados = []
    
    for modelo_info in modelos_disponiveis:
        try:
            models_dir, modelo_file = modelo_info
            
            # Extrair informações do nome do arquivo
            parts = modelo_file.replace('modelo_', '').replace('.joblib', '').split('_')
            ativo = parts[0]
            timeframe = parts[1]
            
            logger.info(f"Processando {ativo} ({timeframe})...")
            
            # Carregar modelo
            modelo_data = joblib.load(os.path.join(models_dir, modelo_file))
            
            # Carregar dados
            dados_file = f"{dados_dir}/{ativo}_{timeframe}_processed.csv"
            if not os.path.exists(dados_file):
                logger.warning(f"Arquivo de dados não encontrado: {dados_file}")
                continue
            
            dados = pd.read_csv(dados_file, index_col=0, parse_dates=True)
            
            # Executar backtest sem lookahead bias
            resultado = executar_backtest_sem_lookahead(dados, modelo_data, f"{ativo}_{timeframe}")
            
            if resultado:
                resultados.append(resultado)
                
                # Salvar resultados detalhados (Prompt 4)
                salvar_resultados_detalhados(resultado, f"{ativo}_{timeframe}")
                
        except Exception as e:
            logger.error(f"Erro no backtest de {modelo_file}: {str(e)}")
            continue
    
    return resultados

def salvar_resultados_detalhados(resultado, ativo_timeframe):
    """
    Salva resultados detalhados do backtest conforme Prompt 4.
    
    Args:
        resultado: Resultado do backtest
        ativo_timeframe: Nome do ativo e timeframe
    """
    try:
        # Salvar estatísticas completas em arquivo de texto
        with open(f'resultados_backtest_{ativo_timeframe}.txt', 'w', encoding='utf-8') as f:
            f.write(f"RESULTADOS DO BACKTEST - {ativo_timeframe}\n")
            f.write("=" * 50 + "\n\n")
            for key, value in resultado.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Estatísticas salvas: resultados_backtest_{ativo_timeframe}.txt")
        
    except Exception as e:
        logger.warning(f"Erro ao salvar resultados detalhados: {e}")

def main():
    """
    Função principal do script de backtesting.
    """
    logger.info("=" * 70)
    logger.info("BACKTESTING DA ESTRATÉGIA DE SWING TRADING")
    logger.info("=" * 70)
    
    inicio = datetime.now()
    logger.info(f"Início: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Executar backtesting usando a biblioteca backtesting (sem lookahead bias)
    resultados = executar_backtesting_completo()
    
    # Gerar relatório
    gerar_relatorio_final(resultados)
    
    fim = datetime.now()
    duracao = fim - inicio
    
    logger.info(f"Fim: {fim.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duração: {duracao}")
    logger.info("=" * 70)

class EstrategiaML(Strategy):
    """
    Estratégia de Machine Learning livre de lookahead bias.
    
    Esta estratégia calcula os indicadores de forma segura usando self.I()
    e faz previsões apenas com dados disponíveis até o momento atual.
    """
    
    def init(self):
        """
        Inicializa os indicadores usando self.I() para evitar lookahead bias.
        """
        # Indicadores técnicos calculados de forma segura
        self.sma_20 = self.I(lambda x: pd.Series(x).rolling(20).mean(), self.data.Close)
        self.sma_50 = self.I(lambda x: pd.Series(x).rolling(50).mean(), self.data.Close)
        
        # RSI - implementação simples
        def rsi_simple(prices, period=14):
            prices_series = pd.Series(prices)
            delta = prices_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.values
        
        self.rsi = self.I(lambda x: rsi_simple(x), self.data.Close)
        
        # MACD - usando implementação simples para evitar problemas de formato
        def macd_simple(prices, fast=12, slow=26, signal=9):
            prices_series = pd.Series(prices)
            ema_fast = prices_series.ewm(span=fast).mean()
            ema_slow = prices_series.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line.values, signal_line.values, histogram.values
        
        macd_values = self.I(lambda x: macd_simple(x), self.data.Close)
        self.macd = self.I(lambda x: x[0] if len(x) == 3 else np.full(len(self.data), np.nan), macd_values)
        self.macd_signal = self.I(lambda x: x[1] if len(x) == 3 else np.full(len(self.data), np.nan), macd_values)
        self.macd_hist = self.I(lambda x: x[2] if len(x) == 3 else np.full(len(self.data), np.nan), macd_values)
        
        # Bollinger Bands - implementação simples
        def bollinger_bands(prices, period=20, std_dev=2):
            prices_series = pd.Series(prices)
            middle = prices_series.rolling(period).mean()
            std = prices_series.rolling(period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper.values, middle.values, lower.values
        
        bb_values = self.I(lambda x: bollinger_bands(x), self.data.Close)
        self.bb_upper = self.I(lambda x: x[0] if len(x) == 3 else np.full(len(self.data), np.nan), bb_values)
        self.bb_middle = self.I(lambda x: x[1] if len(x) == 3 else np.full(len(self.data), np.nan), bb_values)
        self.bb_lower = self.I(lambda x: x[2] if len(x) == 3 else np.full(len(self.data), np.nan), bb_values)
        
        # Stochastic - implementação simples
        def stochastic(high, low, close, k_period=14, d_period=3):
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            
            lowest_low = low_series.rolling(k_period).min()
            highest_high = high_series.rolling(k_period).max()
            k_percent = 100 * ((close_series - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(d_period).mean()
            
            return k_percent.values, d_percent.values
        
        stoch_values = self.I(lambda h, l, c: stochastic(h, l, c), self.data.High, self.data.Low, self.data.Close)
        self.stoch_k = self.I(lambda x: x[0] if len(x) == 2 else np.full(len(self.data), np.nan), stoch_values)
        self.stoch_d = self.I(lambda x: x[1] if len(x) == 2 else np.full(len(self.data), np.nan), stoch_values)
        
        # ATR - implementação simples
        def atr_simple(high, low, close, period=14):
            high_series = pd.Series(high)
            low_series = pd.Series(low)
            close_series = pd.Series(close)
            
            tr1 = high_series - low_series
            tr2 = abs(high_series - close_series.shift(1))
            tr3 = abs(low_series - close_series.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean()
            
            return atr.values
        
        self.atr = self.I(lambda h, l, c: atr_simple(h, l, c), 
                         self.data.High, self.data.Low, self.data.Close)
        
        # Volume Delta (simplificado)
        self.volume_delta = self.I(lambda o, c, v: pd.Series(v) * (pd.Series(c) > pd.Series(o)).astype(int) - 
                                  pd.Series(v) * (pd.Series(c) <= pd.Series(o)).astype(int),
                                  self.data.Open, self.data.Close, self.data.Volume)
        
        # Retorno percentual
        self.retorno_pct = self.I(lambda x: pd.Series(x).pct_change(), self.data.Close)
        
        # Carregar modelo (será definido externamente)
        self.modelo = None
        self.feature_names = None
        self.confianca_minima = 0.45  # Reduzido de 0.6 para 0.45
        
    def next(self):
        """
        Lógica executada a cada step do backtest.
        Calcula features apenas com dados disponíveis até o momento atual.
        """
        # Verificar se temos dados suficientes
        if len(self.data) < 50 or self.modelo is None:
            return
        
        # Verificar se temos indicadores válidos
        if (pd.isna(self.sma_20[-1]) or pd.isna(self.sma_50[-1]) or 
            pd.isna(self.rsi[-1]) or pd.isna(self.atr[-1])):
            return

        try:
            # Coletar valores mais recentes dos indicadores
            features_atuais = [
                self.retorno_pct[-1] if not pd.isna(self.retorno_pct[-1]) else 0,
                self.sma_20[-1] if not pd.isna(self.sma_20[-1]) else self.data.Close[-1],
                self.sma_50[-1] if not pd.isna(self.sma_50[-1]) else self.data.Close[-1],
                self.rsi[-1] if not pd.isna(self.rsi[-1]) else 50,
                self.macd[-1] if not pd.isna(self.macd[-1]) else 0,
                self.macd_signal[-1] if not pd.isna(self.macd_signal[-1]) else 0,
                self.macd_hist[-1] if not pd.isna(self.macd_hist[-1]) else 0,
                self.volume_delta[-1] if not pd.isna(self.volume_delta[-1]) else 0,
                self.bb_upper[-1] if not pd.isna(self.bb_upper[-1]) else self.data.Close[-1] * 1.02,
                self.bb_middle[-1] if not pd.isna(self.bb_middle[-1]) else self.data.Close[-1],
                self.bb_lower[-1] if not pd.isna(self.bb_lower[-1]) else self.data.Close[-1] * 0.98,
                self.stoch_k[-1] if not pd.isna(self.stoch_k[-1]) else 50,
                self.stoch_d[-1] if not pd.isna(self.stoch_d[-1]) else 50,
                self.atr[-1] if not pd.isna(self.atr[-1]) else self.data.Close[-1] * 0.02
            ]
            
            # Organizar em formato que o modelo espera
            features_array = np.array(features_atuais).reshape(1, -1)
            
            # Fazer predição APENAS para o candle atual
            predicao = self.modelo.predict(features_array)[0]
            probabilidades = self.modelo.predict_proba(features_array)[0]
            confianca = max(probabilidades)
            
            # Lógica de entrada - apenas se confiança for alta o suficiente
            if confianca >= self.confianca_minima:
                if predicao == 1:  # Sinal de alta forte
                    if not self.position:  # Se não tem posição, compra
                        self.buy(sl=self.data.Close[-1] * 0.92, tp=self.data.Close[-1] * 1.15)
                elif predicao == 2:  # Sinal de baixa forte  
                    if self.position:  # Se tem posição comprada, vende
                        self.position.close()
                    
        except Exception as e:
            # Log do erro mas continue o backtest
            logger.warning(f"Erro na predição no candle {len(self.data)}: {e}")
            return

def executar_backtest_sem_lookahead(dados, modelo, ativo):
    """
    Executa backtest usando a biblioteca backtesting para eliminar lookahead bias.
    
    Args:
        dados (DataFrame): Dados históricos com indicadores
        modelo: Modelo treinado
        ativo (str): Nome do ativo
    
    Returns:
        dict: Resultados do backtest
    """
    logger.info(f"Executando backtest sem lookahead bias para {ativo}...")
    
    try:
        # Preparar dados no formato esperado pela biblioteca backtesting
        # Colunas necessárias: Open, High, Low, Close, Volume
        dados_bt = dados[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        dados_bt = dados_bt.dropna()
        
        if len(dados_bt) < 100:
            logger.warning(f"Poucos dados para backtest: {len(dados_bt)}")
            return None
        
        # Configurar estratégia com modelo
        class EstrategiaMLComModelo(EstrategiaML):
            def init(self):
                super().init()
                self.modelo = modelo['modelo']
                self.feature_names = modelo.get('feature_names', [])
                self.dados_originais = dados  # Para acessar todas as colunas
        
        # Criar e executar backtest
        bt = Backtest(dados_bt, EstrategiaMLComModelo, cash=10000, commission=0.001)
        
        # Executar backtest
        resultado = bt.run()
        
        # Salvar trades individuais (Prompt 4)
        try:
            trades_df = resultado._trades
            if not trades_df.empty:
                trades_df.to_csv(f'trades_realizados_{ativo}.csv', index=False)
                logger.info(f"Trades salvos: trades_realizados_{ativo}.csv")
        except Exception as e:
            logger.warning(f"Erro ao salvar trades: {e}")
        
        logger.info(f"Backtest concluído para {ativo}")
        logger.info(f"Retorno: {resultado['Return [%]']:.2f}%")
        logger.info(f"Sharpe Ratio: {resultado['Sharpe Ratio']:.2f}")
        logger.info(f"Max Drawdown: {resultado['Max. Drawdown [%]']:.2f}%")
        
        return {
            'ativo': ativo,
            'retorno_total': resultado['Return [%]'] / 100,
            'sharpe_ratio': resultado['Sharpe Ratio'],
            'max_drawdown': abs(resultado['Max. Drawdown [%]']) / 100,
            'total_trades': resultado['# Trades'],
            'win_rate': resultado['Win Rate [%]'] / 100 if resultado['# Trades'] > 0 else 0,
            'profit_factor': resultado['Profit Factor'] if resultado['# Trades'] > 0 else 0,
            'capital_final': resultado['Equity Final [$]']
        }
        
    except Exception as e:
        logger.error(f"Erro no backtest de {ativo}: {e}")
        return None

if __name__ == "__main__":
    main()
