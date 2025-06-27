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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
        print(f"   🔄 Executando backtest para {ativo}...")
        
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

def main():
    """Função principal do backtesting"""
    print("=" * 70)
    print("BACKTESTING DA ESTRATÉGIA DE SWING TRADING")
    print("=" * 70)
    print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configurações
    CAPITAL_INICIAL = 10000  # R$ 10.000
    TAXA_CORRETAGEM = 0.001  # 0.1%
    STOP_LOSS = 0.05         # 5%
    TAKE_PROFIT = 0.10       # 10%
    
    # Diretórios
    dados_dir = "dados_processados"
    
    # Buscar modelos treinados
    modelos_disponiveis = [f for f in os.listdir('.') if f.startswith('modelo_') and f.endswith('.joblib')]
    
    if not modelos_disponiveis:
        print("❌ Nenhum modelo treinado encontrado!")
        print("💡 Execute primeiro o script: python treinamento_modelo.py")
        return
    
    print(f"🤖 Encontrados {len(modelos_disponiveis)} modelos treinados")
    print(f"💰 Capital inicial: R$ {CAPITAL_INICIAL:,.2f}")
    print(f"📊 Configurações:")
    print(f"   • Taxa de corretagem: {TAXA_CORRETAGEM*100:.1f}%")
    print(f"   • Stop Loss: {STOP_LOSS*100:.0f}%")
    print(f"   • Take Profit: {TAKE_PROFIT*100:.0f}%")
    print()
    
    # Inicializar engine de backtesting
    engine = BacktestEngine(
        capital_inicial=CAPITAL_INICIAL,
        taxa_corretagem=TAXA_CORRETAGEM,
        stop_loss=STOP_LOSS,
        take_profit=TAKE_PROFIT
    )
    
    # Executar backtests
    resultados = []
    
    for modelo_file in modelos_disponiveis:
        try:
            # Extrair informações do nome do arquivo
            parts = modelo_file.replace('modelo_', '').replace('.joblib', '').split('_')
            ativo = parts[0]
            timeframe = parts[1]
            
            print(f"📈 Processando {ativo} ({timeframe})...")
            
            # Carregar modelo
            modelo_data = joblib.load(modelo_file)
            modelo = modelo_data['modelo']  # Extrair o modelo do dicionário
            
            # Carregar dados
            dados_file = f"{dados_dir}/{ativo}_{timeframe}_processed.csv"
            if not os.path.exists(dados_file):
                print(f"   ⚠️  Arquivo de dados não encontrado: {dados_file}")
                continue
            
            dados = pd.read_csv(dados_file, index_col=0, parse_dates=True)
            
            # Executar backtest
            resultado = engine.executar_backtest(dados, modelo, f"{ativo}_{timeframe}")
            resultados.append(resultado)
            
            print(f"   ✅ Backtest concluído!")
            print(f"   📊 Retorno: {resultado['retorno_total']*100:+.2f}%")
            print(f"   🎯 Win Rate: {resultado['win_rate']*100:.1f}%")
            print(f"   📈 Total Trades: {resultado['total_trades']}")
            print()
            
        except Exception as e:
            print(f"   ❌ Erro no backtest: {str(e)}")
            continue
    
    # Gerar relatório final
    if resultados:
        print("=" * 70)
        print("RELATÓRIO FINAL DO BACKTESTING")
        print("=" * 70)
        
        # Resumo por ativo
        for resultado in resultados:
            print(f"📊 {resultado['ativo']}:")
            print(f"   💰 Capital Final: R$ {resultado['capital_final']:,.2f}")
            print(f"   📈 Retorno Total: {resultado['retorno_total']*100:+.2f}%")
            print(f"   🎯 Win Rate: {resultado['win_rate']*100:.1f}%")
            print(f"   📊 Total Trades: {resultado['total_trades']}")
            print(f"   🎲 Profit Factor: {resultado['profit_factor']:.2f}")
            print(f"   📉 Max Drawdown: {resultado['max_drawdown']*100:.2f}%")
            print(f"   📈 Sharpe Ratio: {resultado['sharpe_ratio']:.2f}")
            print()
        
        # Estatísticas consolidadas
        capital_total = sum(r['capital_final'] for r in resultados)
        retorno_medio = np.mean([r['retorno_total'] for r in resultados])
        win_rate_medio = np.mean([r['win_rate'] for r in resultados])
        total_trades = sum(r['total_trades'] for r in resultados)
        
        print("📊 CONSOLIDADO:")
        print(f"   💰 Capital Total: R$ {capital_total:,.2f}")
        print(f"   📈 Retorno Médio: {retorno_medio*100:+.2f}%")
        print(f"   🎯 Win Rate Médio: {win_rate_medio*100:.1f}%")
        print(f"   📊 Total de Trades: {total_trades}")
        
        # Performance vs Buy & Hold
        print()
        print("🎯 ANÁLISE DE PERFORMANCE:")
        melhor_resultado = max(resultados, key=lambda x: x['retorno_total'])
        pior_resultado = min(resultados, key=lambda x: x['retorno_total'])
        
        print(f"   🏆 Melhor Performance: {melhor_resultado['ativo']} ({melhor_resultado['retorno_total']*100:+.2f}%)")
        print(f"   📉 Pior Performance: {pior_resultado['ativo']} ({pior_resultado['retorno_total']*100:+.2f}%)")
        
        # Salvar resultados
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv('resultados_backtest.csv', index=False)
        print(f"   💾 Resultados salvos em: resultados_backtest.csv")
        
    else:
        print("❌ Nenhum backtest foi executado com sucesso!")
    
    print()
    print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
