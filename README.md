# Markowitz
Código base para a elaboração de um programa de otimização para o modelo de Markowitz

# Funções disponíveis
## Modelo
### StockData
#### Classe responsável pelo download e tratamento de dados de Equities
    - _download_data
    - get_ticker_data
    - get_column_data
    
### Markowitz
#### Classe responsável pelos cálculos de acordo com a metodologia de Markowitz, o "fit" neste caso irá simular N carteiras e retirar a que representa o maior sharpe
    - _rand_weigths
    - _calculate_returns
    - _cov_matrix
    - _port_vol
    - _port_returns
    - fit
    - _best_sharpe
    - _worst_sharpe
    - _stats
    - backtest_markowitz
