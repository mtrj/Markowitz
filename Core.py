import numpy as np
import scipy.stats.stats as sss
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests as requests
import getpass

# Exemplo para rodar o Markowitz usando dados de 1y diários do yahoo:
# Mark = Markowitz(BaixaAcao(['PETR4','VALE3','BRKM5','VVAR3','TIMP3','ABEV3']).matrizretornos(),np.random.rand(6))
# Neste caso baixamos 6 ações e as transformamos em um objeto da classe de Markowitz para que seja utilizado para qualquer fim,
# os pesos são determinados por uma lista aleatória do numpy
# Caso a simulação seja de fronteira eficiente:
# Mark.PortfolioAleatorio(100000)
# Onde serão gerados 100.000 portfólios aleatórios e plotados

def left(s, amount):
    return s[:amount]
def right(s, amount):
    return s[-amount:]
def mid(s, offset, amount):
    return s[offset:offset+amount]
    
class BaixaAcao:
    def __init__(self, tickerarr):
        self.tickerarr = tickerarr
    
    def download(self):
        tickerarr = self.tickerarr
        precos = []
        for x in range(len(tickerarr)):
            headers = {"User-Agent":'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
            page = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/' + tickerarr[x] + '.SA?region=BR&lang=pt-BR&includePrePost=false&interval=1d&range=1y&corsDomain=br.financas.yahoo.com&.tsrc=finance', headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            adj = str(soup).split()
            adj = str(soup).split('{')
            adj2 = str(adj).split(',')
            adjclosearr =[]
            #Quebras do arquivo:
            for i in range(1536,1784,1):
                if i==1536:
                    numero = float(adj2[1536].replace(left(adj2[1536],14),"")*1)
                    adjclosearr.append(numero)
                elif i!=1536 and i!=1783:
                    numero = float(adj2[i]*1)
                    adjclosearr.append(numero)
                elif i==1783 or i==1784:
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
    
    def PortfolioAleatorio(self, N):
        DIo = 0.049
        RetornosAtivos = self.RetornosAtivos
        NAtivos = len(RetornosAtivos)
        #MatrizResposta = []
        FO = open("C:/Users/" + getpass.getuser() + "/Desktop/TesteMarkowitzAleat.txt",'w+')
        VolArr = []
        RetArr = []
        SharpeArr = []
        for x in range(N):
            Pesos = Utilidades().PesosAleatorios(NAtivos)
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
        plt.style.use('seaborn-whitegrid')
        plt.xlabel('Volatilidade')
        plt.ylabel('Retorno')
        grafico = plt.scatter(VolArr, RetArr, c=SharpeArr, cmap='inferno')
        pontoverm = plt.scatter(VolSM, RetSM,c='red', s=50)
        plt.colorbar(grafico, label='Sharpe')
        plt.savefig("C:/Users/" + getpass.getuser() + "/Desktop/MarkowitzTeste.png")
        print('O sharpe máximo é: ' + left(str(SharpeMax[0]),4) + '\n' + 'O retorno é: ' + left(str(RetSM[0]*100),4) + '%' + '\n' + 'A vol é: ' + left(str(VolSM[0]*100),4) + '%')
