# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from datetime import datetime as dt
import yfinance as yf
import time
from lib_yahoo import yahoo
inicio = date(2019,5,22)
#inicio_timestamp = int(dt.timestamp(dt(2019,6,26)))

class StockData:
    """ Classe responsável por download e manipulação dos dados do yfinance """
    def __init__(self,tickers, end_date=None):
        if end_date is None: end_date = date.today()
        self.end_date = end_date
        self.tickers = tickers
        self._complete_data = {}
        self._download_data()
        
    def _download_data(self):
        for ticker in self.tickers:
            yf_tk = ticker + '.sa'
            self._complete_data[ticker] = yf.download(tickers=yf_tk,start=inicio,end=self.end_date,progress=False)
        
    def get_ticker_data(self, ticker):
        """Retorna as todos os dados para um ticker"""
        return self._complete_data[ticker]
    
    def get_column_data(self, column):
        """Retorna um df com os valores de um atributo para todos os tickers"""
        dtf = pd.DataFrame()
        for ticker in self.tickers:
            dtf2 = pd.DataFrame({ticker:self._complete_data[ticker][column]})
            dtf = pd.concat([dtf,dtf2],axis=1)
        return dtf
   
class Markowitz:
    """Classe para cálculo de portifólios ótimo"""
    def __init__(self, quotes,n_port=10000):
        self.dio = 0.0415
        self.quotes = quotes
        self.tickers = list(quotes.columns)
        self.returns = np.array([0])
        self._avg_rets = np.array([0])
        self.n_port = n_port
        self._cov_mat = np.zeros(shape=(len(self.tickers),len(self.tickers)))
        self._calculate_returns()  
        self._cov_matrix()
    
    def _rand_weigths(self):
        ws = np.random.rand(len(self.tickers))
        return ws/sum(ws)
        
    def _calculate_returns(self):
        returns = self.quotes/self.quotes.shift(1)-1
        returns = returns.dropna()
        self.returns = np.array(returns)
        self._avg_rets = self.returns.mean(axis=0)*252
        self._avg_vol = self.returns.std(axis=0)*np.sqrt(252)
        
    def _cov_matrix(self):
        self._cov_mat = np.cov(self.returns,rowvar=False)*252
        
    def _port_vol(self,pesos):
        return np.sqrt(np.matmul(np.matmul(pesos,self._cov_mat),pesos.T))
    
    def _port_returns(self,pesos):
        return np.matmul(pesos,self._avg_rets.T)
        
    def fit(self, sg=True):
        start_time = time.time()
        retorn_arr, vol_arr, sharpe_arr, pesos_arr = [], [], [], []
        for i in range(self.n_port):
            pesos = self._rand_weigths()
            ret = self._port_returns(pesos)
            vol = self._port_vol(pesos)
            sharpe = (ret-self.dio)/vol
            
            pesos_arr.append(pesos)
            retorn_arr.append(ret)
            vol_arr.append(vol)
            sharpe_arr.append(sharpe)
            
        sharpe_max = np.max(sharpe_arr)
        sharpe_max_loc = sharpe_arr.index(sharpe_max)
        vol_sm = vol_arr[sharpe_max_loc]
        ret_sm = retorn_arr[sharpe_max_loc]
        pesos_sm = pesos_arr[sharpe_max_loc]
        if sg==True:
            plt.clf()
            plt.style.use('seaborn-whitegrid')
            plt.xlabel('Volatilidade')
            plt.ylabel('Retorno')
            grafico = plt.scatter(vol_arr, retorn_arr, c=sharpe_arr, cmap='inferno')
            plt.colorbar(grafico, label='Sharpe')
            plt.scatter(vol_sm, ret_sm,c='red', s=50)
            plt.show()
            elapsed_time = time.time() - start_time
            print('Tempo de execução: ' + str(round(elapsed_time,4)) + 's')
            for x in range(len(pesos_sm)):
                print('Ativo ' + str(tickers[x].upper()) + ': ' + str(round(pesos_sm[x]*100,4)) + '%')
        else:
            return pesos_sm

    def _best_sharpe(self, n=5,sd=False):
        """Retorna o número n de ativos de maior sharpe na lista tickers"""
        sharpe = [(self._avg_rets[i]-self.dio)/self._avg_vol[i] for i in range(len(self.tickers))]
        sp_backup = list(sharpe)
        sharpe.sort()
        if sd==False: return [self.tickers[sp_backup.index(sharpe[-i])] for i in range(n)]
        if sd==True: return sharpe
    
    def _worst_sharpe(self, n=5,sd=False):
        """Retorna o número n de ativos de maior sharpe na lista tickers"""
        sharpe = [(self._avg_rets[i]-self.dio)/self._avg_vol[i] for i in range(len(self.tickers))]
        sp_backup = list(sharpe)
        sharpe.sort()
        if sd==False: return [self.tickers[sp_backup.index(sharpe[i])] for i in range(n)]
        if sd==True: return sharpe
        
    def _stats(self, plot=[], inicio_plot=50):
        """
        Função para plotar as estatísticas dos dados de ações
        """
        df_out = pd.DataFrame(index=self.quotes.index)
        for i in range(len(self.tickers)):
            ticker=self.tickers[i]
            rets_hist = [np.log(self.quotes[:i][ticker]/self.quotes[:i][ticker].shift(1)).mean()*252 for i in range(len(self.quotes))]
            vols_hist = [np.log(self.quotes[:i][ticker]/self.quotes[:i][ticker].shift(1)).std()*np.sqrt(252)
                         for i in range(len(self.quotes))]
            sps_hist = [(rets_hist[i]-self.dio)/vols_hist[i] for i in range(len(self.quotes))]
            df_out[ticker+'_ret'] = rets_hist
            df_out[ticker+'_vol'] = vols_hist
            df_out[ticker+'_sharpe'] = sps_hist
        df_out[[self.tickers[i]+'_vol' for i in range(len(self.tickers))]][inicio_plot:].plot()
        
def backtest_markowitz(tickers, ndays, plot=False, ibov_plot=False, override=True, position='long'):
    """
    Gera a carteira com o melhor sharpe a cada ndays e computa qual a porcentagem
    deve ser alocada no ativo, posteriormente faz o PnL baseado nisso
    """
    #Ainda não tem troca de ativos no meio
    if override==False:
        data = StockData(tickers)
        dataframe = data.get_column_data('Adj Close').dropna()
    else:
        dataframe = yahoo(tickers)._consolidate_dfs()
        dataframe = dataframe[dataframe.index.get_loc(inicio):]
    df=dataframe[ndays:]
    start_time = time.time()
    pesos_aleat = []
    df_sliced = df.iloc[::ndays, :].dropna()
    ret_close = np.log(df_sliced/df_sliced.shift(1)).dropna()
    dt_recalc = df_sliced.index.to_pydatetime()
    dr_list = dataframe.index.to_list()
    for i in range(len(ret_close)):
        close = dataframe[:dr_list.index(dt_recalc[i])]
        mark = Markowitz(close)
        pesos_aleat.append(mark.fit(sg=False))
    pesos_aleat = pd.DataFrame(pesos_aleat)
    if position.lower()=='long': avg_close = np.array(pesos_aleat)*np.array(ret_close)
    if position.lower()=='short': avg_close = -np.array(pesos_aleat)*np.array(ret_close)
    pnl = pd.DataFrame(np.cumsum([np.sum(avg_close[i]) 
                                  for i in range(len(avg_close))])+1,
                                  index=dt_recalc[1:], columns={'Pnl'})
    if plot==True: pnl.plot(color='black')
    if ibov_plot==True:
        ibov = yf.download('^BVSP',start=inicio,progress=False)['Adj Close'].dropna()
        ibov['ret'] = np.log(ibov/ibov.shift(1))
        ibov['IBOVESPA'] = ibov['ret'].cumsum()+1
        ibov['IBOVESPA'].plot(color='blue')
    elapsed_time = time.time() - start_time
    print('Tempo de execução: ' + str(round(elapsed_time,4)) + 's')
    return pnl
            
tickers= ['abev3','vvar3','jbss3','petr4', 'btow3', 'itub4']
