import numpy as np
import scipy.stats.stats as sss
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests as requests
import getpass
from itertools import combinations

# Exemplo para rodar o Markowitz usando dados de 1y diários do yahoo:
# Acoes = ['PETR4','VALE3','BRKM5','VVAR3','TIMP3','ABEV3']
# tenor = '2y' # ou seja, o modelo utilizará dados de 2 anos
# Mark = Markowitz(BaixaAcao(Acoes,tenor).matrizretornos(),np.random.rand(len(Acoes)))
# Neste caso baixamos 6 ações e as transformamos em um objeto da classe de Markowitz para que seja utilizado para qualquer fim,
# os pesos são determinados por uma lista aleatória do numpy
# Caso a simulação seja de fronteira eficiente:
# Mark.PortfolioAleatorio(100000,Acoes) #Aqui tem que colocar a matriz com os tickers para ele gerar a resposta com a composição ideal
# Onde serão gerados 100.000 portfólios aleatórios e plotados

def left(s, amount):
    return s[:amount]
def right(s, amount):
    return s[-amount:]
def mid(s, offset, amount):
    return s[offset:offset+amount]
    
class BaixaAcao:
    def __init__(self, tickerarr, tenor='1y'):
        self.tickerarr = tickerarr
        self.tenor = tenor
        
    def download(self):
        tickerarr = self.tickerarr
        tenor = self.tenor
        precos = []
        for x in range(len(tickerarr)):
            headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
            page = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/' + tickerarr[x] + '.SA?region=BR&lang=pt-BR&includePrePost=false&interval=1d&range=' + tenor + '&corsDomain=br.financas.yahoo.com&.tsrc=finance', headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            adj = str(soup).split()
            adj = str(soup).split('{')
            adj2 = str(adj).split(',')
            adjclosearr =[]
            #   Quebras do arquivo do Yahoo, 45 linhas de cabeçalho, 3 linhas de cabeçalho a mais
            # até chegar no adjusted close, uma linha final com null, somando 49 e 48 antes de adjclose:
            inicio = int(((len(adj2)-49)/7*6+48))
            fim = len(adj2) - 1
            for i in range(inicio,fim,1):
                if i==inicio:
                    numero = float(adj2[inicio].replace(left(adj2[inicio],14),"")*1)
                    adjclosearr.append(numero)
                elif i!=inicio and i!=(fim-1):
                    numero = float(adj2[i]*1)
                    adjclosearr.append(numero)
                elif i==(fim-1) or i==fim:
                    numero = float(adj2[i].replace(right(adj2[i],6),"")*1)
                    adjclosearr.append(numero)
            precos.append(adjclosearr)
        if len(tickerarr)==1:
            return(adjclosearr)
        elif len(tickerarr)>1:
            return(precos)
    
    def matrizretornos(self):
        tickerarr = self.tickerarr
        matrizprecos = BaixaAcao(tickerarr).download()
        RA = [Precos(matrizprecos[i]).Retornos() for i in range(len(tickerarr))]
        return(RA)
        
class Precos:
    def __init__(self,Precos):
        self.Precos = Precos
    
    def Retornos(self):
        Precos = self.Precos
        Retornos = []
        for i in range(len(Precos)-1):
            Retornos.append(np.log(Precos[i+1]/Precos[i]))
        return(Retornos)
        
    def ERetorno(self):
        Precos = self.Precos
        Retornos = []
        for i in range(len(Precos)-1):
            Retornos.append(np.log(Precos[i+1]/Precos[i]))
            #print(Retorno[i])
        return(np.average(Retornos))
    
    def Volatilidade(self):
        Precos = self.Precos
        Retornos = []
        for i in range(len(Precos)-1):
            Retornos.append(np.log(Precos[i+1]/Precos[i]))
        #ddof=1 é stdev de sample
        return(np.std(Retornos, ddof=1, dtype=np.float64))
    
class Correlacao:
    def __init__(self, RetornosAtivos):
        self.RetornosAtivos = RetornosAtivos
    #def Matriz(self):
    #    RetornosAtivos = self.RetornosAtivos
    #    NAtivos = len(RetornosAtivos)
    #    MatrizCorrel = []
    #    for i in range(NAtivos):
    #        Linha = []
    #        for x in range(NAtivos):
    #            Linha.append(sss.pearsonr(RetornosAtivos[i],RetornosAtivos[x])[0])#sss.pearsonr(RetornosAtivos[i],RetornosAtivos[x])[0])
            #print(Linha)
    #        MatrizCorrel.append(Linha)
    #    return(MatrizCorrel)
    
    def Matriz(self):
        RetornosAtivos = self.RetornosAtivos
        return(np.cov(RetornosAtivos)*252)
        
class Utilidades:
    def __init__(self, Matriz=[]):
        self.Matriz = Matriz
    
    def Transpose(self):
        Matriz = self.Matriz
        mr = [[Matriz[j][i] for j in range(len(Matriz))] for i in range(len(Matriz[0]))] 
        MatrizTransposta = [row for row in mr]
        return(MatrizTransposta)
    
    def PesosAleatorios(self, NAtivos):
        MatrizAleatoria = [np.random.rand() for x in range(NAtivos)]
        Soma = sum(MatrizAleatoria)
        MatrizPesos = [MatrizAleatoria[x]/Soma for x in range(NAtivos)]
        return(MatrizPesos)

class Markowitz:
    def __init__(self,RetornosAtivos,Pesos):
        self.RetornosAtivos = RetornosAtivos
        self.Pesos = Pesos
    
    def ERetorno(self):
        RetornosAtivos = self.RetornosAtivos
        Pesos = self.Pesos
        MatrizRetornos = []
        for i in range(len(RetornosAtivos)):
            MatrizRetornos.append(np.average(RetornosAtivos[i])*252)
        RetornosTransp = Utilidades([MatrizRetornos]).Transpose()
        return(np.matmul(Pesos,RetornosTransp)[0])
    
    def Variancia(self):
        RetornosAtivos = self.RetornosAtivos
        Pesos = self.Pesos
        MatrizCorrelacao = Correlacao(RetornosAtivos).Matriz()
        PesosTransp = Utilidades(Pesos).Transpose()
        return(np.matmul(np.matmul(Pesos,MatrizCorrelacao),PesosTransp)[0])
    
    def Volatilidade(self):
        RetornosAtivos = self.RetornosAtivos
        Pesos = self.Pesos
        PesosTransp = Utilidades(Pesos).Transpose()
        MatrizCorrelacao = Correlacao(RetornosAtivos).Matriz()
        return(np.sqrt(np.matmul(np.matmul(Pesos,MatrizCorrelacao),PesosTransp)[0]))
    
    def PortfolioAleatorio(self, N, tickerarr):
        DIo = 0.049
        RetornosAtivos = self.RetornosAtivos
        NAtivos = len(RetornosAtivos)
        #MatrizResposta = []
        FO = open("C:/Users/" + getpass.getuser() + "/Desktop/Markowitz/Markowitz" + str(tickerarr) + ".txt",'w+')
        PesosArr =[]
        VolArr = []
        RetArr = []
        SharpeArr = []
        for x in range(N):
            Pesos = Utilidades().PesosAleatorios(NAtivos)
            PesosArr.append(Pesos)
            ObjMarkowitz = Markowitz(RetornosAtivos, [Pesos])
            VarPort = ObjMarkowitz.Variancia()
            VolPort = np.sqrt(VarPort)
            VolArr.append(VolPort)
            RetPort = ObjMarkowitz.ERetorno()
            RetArr.append(RetPort)
            SharpeArr.append((RetPort - DIo)/VolPort)
            for k in range(len(Pesos)):
                FO.write(str(Pesos[k]) + '|')
            FO.write(str(VolPort[0]) + '|' + str(RetPort[0]))
            FO.write('|' + str(float(SharpeArr[x])))
            FO.write('\n')
        FO.close()
        SharpeMax = max(SharpeArr)
        SharpeMaxLoc = SharpeArr.index(SharpeMax)
        VolSM = VolArr[SharpeMaxLoc]
        RetSM = RetArr[SharpeMaxLoc]
        plt.clf()
        plt.style.use('seaborn-whitegrid')
        plt.xlabel('Volatilidade')
        plt.ylabel('Retorno')
        grafico = plt.scatter(VolArr, RetArr, c=SharpeArr, cmap='inferno')
        pontoverm = plt.scatter(VolSM, RetSM,c='red', s=50)
        plt.colorbar(grafico, label='Sharpe')
        plt.savefig("C:/Users/" + getpass.getuser() + "/Desktop/Markowitz/Markowitz" + str(tickerarr) + ".png")
        #print('O sharpe máximo é: ' + left(str(SharpeMax[0]),4) + '\n' + 'O retorno é: ' + left(str(RetSM[0]*100),4) + '%' + '\n' + 'A vol é: ' + left(str(VolSM[0]*100),4) + '%' + '\n')
        #print('Composição ideal da carteira:')
        #for v in range(len(RetornosAtivos)):
            #print(tickerarr[v] + ': ' + left(str(PesosArr[SharpeMaxLoc][v]*100),6) + '%')
        return([SharpeMax[0],RetSM[0],VolSM[0],PesosArr[SharpeMaxLoc]])
            
    def CarteirasOtimas(self,AtivosCarteira, N, tickerarr):
        Nativos = len(tickerarr)
        pesosaleat = np.random.rand(Nativos)
        arr = [i for i in range(Nativos)]
        r = AtivosCarteira
        comb = combinations(arr,r)
        combarr = []
        carteiraarr = []
        resposta =[]
        ResultArr = []
        sharpearr = []
        for w in list(comb):
            combarr.append(w)
        for i in range(len(combarr)):
            carteira = []
            for x in range(len(combarr[i])):
                numero = combarr[i][x]
                carteira.append(tickerarr[numero])
            carteiraarr.append(carteira)
        for z in range(len(carteiraarr)):
            RetornosAtivos = BaixaAcao(carteiraarr[z],'2y').matrizretornos()
            ResultArr.append(Markowitz(RetornosAtivos, pesosaleat).PortfolioAleatorio(N, carteiraarr[z]))
            resposta.append([carteiraarr[z],ResultArr[z]])
            sharpearr.append(ResultArr[z][0])
        MaxSharpe = max(sharpearr)
        MaxSharpeLoc = sharpearr.index(MaxSharpe)
        return(resposta[MaxSharpeLoc])
# http://wilsonfreitas.github.io/posts/expectativas-do-copom-nos-contratos-de-di1.html
# Para projetar todas as carteiras possíveis em um array de ativos:
# from itertools import combinations
#arr = [i for i in range(10)] #gera numeros de 0 a 9 (10 ativos)
#r = 3 #numero de quebra de ativos por carteira
#perm = combinations(arr, r) #gera o objeto a ser trabalhado
#permarr = [] #inicia a matriz que irá armazenar os valores
#for i in list(perm):
#	permarr.append(i)

#o resultado será uma matriz de m x n, onde n = r e m = n!/(n-r)!
