# -*- coding: utf-8 -*-
"""
@author: Milton Rocha
"""

import pandas as pd
import numpy as np
import pdblp as bbg
from utils import feriados
import scipy.optimize as solver
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.stats import norm

fer = feriados()

# ============================================================
# Inicia a conexão com a BBG para puxar os dados de equities :
con = bbg.BCon(debug=False, timeout=8000)
con.start()
# ============================================================

# 4Y intervalo de dados :
today = np.datetime64('today').astype('datetime64[D]')
yest = np.busday_offset(today, -1, holidays=fer)
begin = np.busday_offset(today, -1008, holidays=fer)
today_trat = today.astype(str).replace('-','')
yest_trat = yest.astype(str).replace('-','')
begin_trat = begin.astype(str).replace('-','')

cdi = con.bdh('BZDIOVRA Index', 'PX_LAST', yest_trat, yest_trat).values[0][0]/100.0

tickers = ['VALE3 BZ Equity', 'PETR4 BZ Equity', 'GGBR4 BZ Equity', 'CASH3 BZ Equity', 'PRIO3 BZ Equity']
tickers.sort()

# ========================================================
# Puxa os dados dos tickers e o DI (cálculo de Sharpe) :
dados = con.bdh(tickers, 'PX_LAST', begin_trat, today_trat)
dados = dados.dropna()
# ========================================================

class markowitz:
    
    """Classe para cálculo de portifólios ótimos"""
    
    def __init__(self, quotes, dio):
        self.dio = dio
        self.quotes = quotes
        self.tickers = list(quotes.columns)
        self.returns = np.array([0])
        self._avg_rets = np.array([0])
        self._cov_mat = np.zeros(shape=(len(self.tickers),len(self.tickers)))
        self._calculate_returns()  
        self._cov_matrix()
    
    def _rand_weights(self):
        ws = np.random.rand(len(self.tickers))
        return ws/sum(ws)
        
    def _calculate_returns(self):
        returns = self.quotes.pct_change()
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
    
    def _sharpe(self, pesos):
        return (self._port_returns(pesos) - self.dio)/self._port_vol(pesos)
    
    def _sharpe_otim(self, pesos):
        return -(self._port_returns(pesos) - self.dio)/self._port_vol(pesos)
    
    def fit(self,
            pesoMin = None, pesoMax = None,
            retMin = None, retMax = None):
        
        x0 = self._rand_weights()
        if pesoMin != None and pesoMin * len(self.tickers) > 1.0: raise Exception(f'Solução impossível com o pesoMin definido : {pesoMin}')
        if pesoMax != None and pesoMax * len(self.tickers) < 1.0: raise Exception(f'Solução impossível com o pesoMax definido : {pesoMax}')
        if pesoMin == None: pesoMin = 0
        if pesoMax == None: pesoMax = 1
        if retMin == None: retMin = min(self._avg_rets)
        if retMax == None: retMax = max(self._avg_rets)
        
        bounds = tuple((pesoMin, pesoMax) for x in range(len(self.tickers)))
        
        constraints = [{'type' : 'eq', 'fun' : lambda x : sum(x) - 1},
                        {'type' : 'ineq', 'fun' : lambda x : retMax - sum(x * self._avg_rets)},
                        {'type' : 'ineq', 'fun' : lambda x : sum(x * self._avg_rets) - retMin}]
        
        otim = solver.minimize(self._sharpe_otim, x0 = x0, constraints = constraints, bounds = bounds, method = 'SLSQP', options={'disp' : False})
        self.peso_otim = otim['x']
        self.vol_otim = self._port_vol(self.peso_otim)
        self.ret_otim = self._port_returns(self.peso_otim)
        return otim['x']
    
    def _efficient_frontier(self):
        pesos, riscos = [], []
        faixa_ret = np.arange(min(self._avg_rets), max(self._avg_rets), (max(self._avg_rets)-min(self._avg_rets))/1000)
        x0 = self._rand_weights()
        for i in trange(len(faixa_ret)):
            ret = faixa_ret[i]
            constraints = [{'type' : 'eq', 'fun' : lambda x : sum(x) - 1},
                           {'type' : 'eq', 'fun' : lambda x : sum(x * self._avg_rets) - ret}]
            bounds = tuple((0, 1) for x in range(len(self.tickers)))
            otim = solver.minimize(self._sharpe_otim, x0 = x0, constraints = constraints, bounds = bounds, method = 'SLSQP')
            pesos.append(otim['x']), riscos.append(self._port_vol(otim['x']))
        
        plt.clf()
        plt.plot(riscos, faixa_ret, '--', color='black')
        self.fit()
        plt.plot(self.vol_otim, self.ret_otim, 'o', color='red')
        return pesos, riscos
    
    def VaR(self, port_value=1000000, alfa=0.05, n_dias=1):
        try: self.peso_otim*1
        except: self.fit()
        return np.sum(((self._avg_rets/252.0 + norm.ppf(alfa) * self._avg_vol/np.sqrt(252)))*self.peso_otim*port_value)*np.sqrt(n_dias)
    
    def stats(self,
            pesoMin = None, pesoMax = None,
            retMin = None, retMax = None,
            plot = False):
        
        self.fit(pesoMin = pesoMin, pesoMax = pesoMax, retMin = retMin, retMax = retMax)
        print('\n---Estatísticas do Portfólio (anualizadas):\n')
        print('--Composição da carteira ótima :')
        print('\n+++++++++++++++++++++++++\n')
        for i in range(len(self.tickers)):
            print(f'- {self.tickers[i][0]} : {self.peso_otim[i]*100:.2f}%')
        print('\n+++++++++++++++++++++++++\n')
        print(f'-Ret : {self.ret_otim*100:.2f}%')
        print(f'-Vol : {self.vol_otim*100:.2f}%')
        print(f'-Sharpe : {self._sharpe(self.peso_otim):.2f}\n')
        if plot==True: self._efficient_frontier()
