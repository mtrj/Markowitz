import numpy as np
import scipy.stats.stats as sss
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests as requests

# Exemplo para rodar o Markowitz usando dados de 1y diários do yahoo:
# Mark = Markowitz(BaixaAcao(['PETR4','VALE3','BRKM5','VVAR3','TIMP3','ABEV3']).matrizretornos(),np.random.rand(6))
# Neste caso baixamos 6 ações e as transformamos em um objeto da classe de Markowitz para que seja utilizado para qualquer fim,
# os pesos são determinados por uma lista aleatória do numpy
# Caso a simulação seja de fronteira eficiente:
# Mark.PortfolioAleatorio(100000)
# Onde serão gerados 100.000 portfólios aleatórios e plotados, caso o usuário queira salvar o arquivo com os resultados e o gráfico
# deve mudar o caminho destes no código do PortfolioAleatorio.

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
        FO = open("C:/Users/milto/Desktop/TesteMarkowitzAleat.txt",'w+')
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
            #LinhaPesos = [Pesos[k] for k in range(len(Pesos))]
            #LinhaRisco = [str(VolPort[0]) + '|' + str(RetPort[0])]
            #MatrizResposta.append(str(LinhaPesos) + '|' + str(LinhaRisco))
        #for y in range(len(MatrizResposta)):
            #FO.write(str(MatrizResposta[y]))
            #FO.write('\n')
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
        plt.savefig("C:/Users/milto/Desktop/MarkowitzTeste.png")
        print('O sharpe máximo é: ' + left(str(SharpeMax[0]),4) + '\n' + 'O retorno é: ' + left(str(RetSM[0]*100),4) + '%' + '\n' + 'A vol é: ' + left(str(VolSM[0]*100),4) + '%')
#Para chegar na resposta:
#
#
# TesteX = Correlacao(RA)
# Pesos = [[0.50,0.25,0.25]]
# PesosTransp = Utilidades(Pesos).Transpose()
# MatCorrel = TesteX.Matriz()
#
# Var = np.matmul(np.matmul(Pesos,MatCorrel),PesosTransp)
        
#class Respostas:
    #def __init__(self):
        
    #def RetornoEsperado(self):
        
    #def Volatilidade(self):
    

#PrecosMGLU = [18.518934,19.086765,19.367149,18.86841,20.133005,20.285698,20.42761,19.639135,19.340906,19.659369,19.672539,19.722639,19.605734,19.149942,19.610409,20.041103,19.981457,20.351263,20.288088,20.983511,21.115879,21.22687,21.801857,21.086102,20.756857,21.600302,22.783854,22.540028,21.854027,22.193306,22.014553,21.941051,22.154203,21.756029,21.927528,21.678902,21.437529,21.373827,21.339527,20.457525,19.8512,20.776028,20.516325,20.504076,21.817278,21.578354,21.936153,21.903028,21.691153,21.498777,20.788275,20.433027,20.825026,20.252998,20.188026,19.759275,20.909599,20.086399,20.646175,20.783377,19.906275,19.734776,21.792778,21.278276,21.866278,21.560026,21.248878,21.464478,21.022301,21.700853,22.15048,22.364902,21.909203,21.716827,21.621279,21.705851,22.101479,22.135778,22.249754,22.325603,21.229277,21.206051,21.632254,20.580027,20.921753,21.174202,20.983002,20.353352,20.711151,20.702526,20.823851,20.425676,20.372952,20.665777,20.335026,20.19175,20.028776,20.298515,19.939207,21.309076,20.859938,21.334028,21.516178,21.833069,22.325823,22.270927,23.861673,23.796799,24.484179,24.978329,23.97266,24.141136,23.192957,23.267815,22.459368,22.238491,21.583549,20.834986,21.062052,21.583549,22.383312,22.488012,22.512964,22.063927,22.20735,23.579716,23.613451,24.064983,24.453037,24.295841,24.268393,23.852989,23.705673,24.758652,25.263981,25.849157,25.341331,26.151077,26.395607,25.887783,25.962639,26.199684,26.530249,26.197187,25.764219,26.006254,26.031206,26.339415,26.224634,26.506596,27.553385,27.584524,28.205832,28.620035,29.131552,28.791006,28.950701,28.775936,29.193933,30.490145,30.815819,29.699263,30.859535,30.192017,30.667303,30.501421,31.439619,32.911793,33.00531,32.901814,34.49625,35.375759,34.433868,36.529842,36.240402,37.06881,37.707581,36.350189,37.777447,36.879173,36.240402,37.488007,37.757488,36.459976,36.469959,34.364002,33.715248,33.635403,34.144424,34.843082,37.188576,36.22044,36.539825,35.811226,37.128696,36.629654,35.930996,34.134445,32.437702,34.533676,34.932911,34.034634,34.803158,34.783195,34.992798,35.74136,36.489922,35.921017,36.69952,36.649616,36.499901,36.729462,36.969002,37.378216,36.430035,37.2285,38.705666,38.326397,38,39.650002,38.939999,40.5,42.169998,41.799999,42.759998,43.169998,43,43.099998,43.720001,43.490002,43.25,42,42.669998,41.150002,44.02,44.639999,47.189999,44.970001,44.040001,45.200001,44.200001,43.810001,45.02,43.400002,42.5,44.25,44.98,44.52]
#PrecosVvar = [5.08,4.92,4.95,4.95,4.95,4.95,5.04,5.18,5.12,5.15,5.2,5.18,5.04,4.87,4.95,4.78,4.7,4.35,4.36,4.58,4.66,4.58,4.56,4.3,4.32,4.39,4.38,4.31,4.27,4.24,4.04,4.01,4.42,4.51,4.82,4.98,5.05,4.93,4.88,4.8,4.93,5.16,5.49,5.35,5.67,5.77,6,5.96,5.99,5.79,5.29,5.45,5.56,5.52,5.56,5.45,5.45,5.28,5.47,5.28,5.21,4.67,4.72,4.97,4.77,4.64,4.58,4.54,4.39,4.41,4.68,4.76,4.84,4.62,4.64,4.54,4.59,4.64,4.58,4.55,4.33,4.33,4.36,4.11,4.18,4.22,4.27,4.55,4.38,4.38,4.53,4.37,4.34,4.31,4.29,4.16,4.15,4.01,3.92,3.95,3.84,3.93,3.93,4.02,4.03,4.12,4.1,4.14,4.52,4.56,4.72,4.53,4.74,4.61,4.73,4.55,4.3,4.09,4.03,4.11,4.31,4.32,4.2,4.2,4.26,4.41,4.58,4.8,4.7,4.96,4.86,4.65,4.95,5,5.06,5,4.84,5.04,4.97,5.07,5.07,5.06,5.1,5.11,5.04,4.98,4.97,5.07,5.12,5.45,5.99,6.24,6.35,6.75,6.73,6.5,6.52,7.03,7.13,7.31,7.38,7.15,7.05,6.92,7.32,7.1,7.34,7.68,7.62,7.72,8.2,8.19,7.92,8.18,8.44,8.53,8.53,8.28,8.46,8.11,7.69,7.12,6.73,7.06,7.18,6.9,6.79,6.48,6.85,7,7.31,7.73,7.79,7.56,7.62,7.55,7.38,7.01,6.78,6.98,7.1,6.91,6.95,6.97,7.03,7.4,7.4,7.56,7.47,7.64,7.86,7.87,7.92,7.8,7.47,7.8,7.89,7.83,7.72,7.59,7.53,7.75,7.78,7.79,7.82,7.9,7.84,7.81,7.81,7.65,7.49,7.43,7.36,7.35,7.4,7.42,7.64,7.57,7.38,7.45,7.42,7.25,7.31,7.09,7.02,7.6,7.56,7.45]
#PrecosPetro = [23.782598,23.85849,23.118547,23.004707,24.218975,24.076677,23.877464,24.152567,24.60792,24.038731,24.247435,23.327248,23.498003,22.236303,22.094006,22.103491,22.160412,21.866327,21.695574,20.870249,21.10741,20.386438,20.443357,21.274069,21.264257,22.255346,23.609507,24.188459,24.257149,24.639847,24.492655,25.00292,24.787039,24.522095,24.384716,24.365091,24.355276,24.688911,24.924417,25.051985,24.659472,24.953856,25.061798,24.306213,24.894981,25.140299,25.101048,25.316925,25.542622,25.591684,25.012732,24.620222,24.659472,24.374903,25.238426,25.562244,26.445396,26.337456,26.258953,26.867342,26.53371,26.886969,26.622023,26.200077,26.082325,26.572962,26.553335,26.200077,26.258953,26.268764,26.170639,27.230415,26.985098,27.573862,27.662178,27.721054,28.201881,28.653267,28.427574,28.025249,26.494459,26.828093,28.093939,26.828093,27.534613,27.534613,27.475737,27.760307,27.024347,27.936935,28.241131,28.702332,28.614017,28.241131,27.475737,25.346365,25.444494,26.219702,26.24914,27.083225,26.926224,27.161726,27.083225,27.27948,26.758541,26.876379,26.621069,26.247921,26.365759,26.287199,25.874775,26.876379,26.346117,26.198824,25.43289,25.531088,25.41325,24.814251,24.234892,25.059744,26.012249,25.913681,25.470123,25.716545,25.864395,26.41638,26.120676,25.775684,25.184273,25.617975,25.82497,25.489838,25.903824,26.376953,26.268526,26.771227,26.465666,26.790941,26.672657,26.721943,27.057076,27.126074,27.875196,27.845625,27.116219,27.273928,26.840225,27.017649,26.869797,26.436094,26.790941,26.997934,27.007793,27.254213,27.668201,27.993477,28.121616,27.776628,27.431637,27.283783,27.116219,27.04722,27.096504,27.126074,26.958509,26.505093,25.765827,26.002392,25.864395,25.706686,25.874254,26.140388,25.184273,25.509548,25.233559,25.972822,25.903824,25.282843,25.620609,24.756323,24.070854,23.752956,23.872168,23.862234,25.282843,25.054352,24.120527,23.802626,24.180132,24.42849,25.332512,25.332512,25.133825,25.431856,26.087521,26.216667,26.345814,26.75312,26.922005,26.693516,26.882265,26.703447,27.875698,27.508129,27.041214,27.110756,26.82266,27.299507,27.090887,27.160427,27.518063,27.478325,27.369047,27.329311,26.544498,26.564367,26.335878,25.998112,25.849098,26.345814,26.564367,27.080954,27.130623,27.41872,27.746552,27.478325,27.41872,27.587603,28.382349,28.759853,28.133989,29.057882,29.405584,29.624138,29.882431,30.190393,30.230131,30.160591,29.455254,29.51486,30.697044,29.822824,30.250002,30.02,29.9,29.299999,29.08,29.290001]
#PrecosVale = [54.529999,54.040001,50.349998,50.119999,49.919998,52.299999,52,52.799999,54.110001,52.880001,52.799999,52.299999,51.290001,50.200001,50.599998,50.369999,50.599998,50.880001,51.25,51.439999,50.009998,49.810001,50.860001,50.439999,49.5,51,51.09,49,52.189999,51.91,52.41,53.689999,53.099998,52.380001,52.599998,52.349998,52.650002,54.23,54.759998,55.279999,55.080002,55.650002,56.150002,42.360001,42.740002,46.599998,45.5,46.25,44.68,44.52,42.459999,41.59,43.16,42.02,44.299999,45.490002,45.66,45.880001,45.25,45.490002,45.799999,45.380001,46.990002,47.119999,47.200001,46.830002,47.099998,46.740002,48.049999,48.860001,48.849998,49.880001,49.970001,50.700001,50.709999,50.549999,50.459999,51.900002,50.560001,50.889999,50.049999,49.549999,50.279999,49.599998,49.299999,50.93,52.599998,51.630001,51.779999,52.16,51.98,53.389999,52.349998,51.759998,51.790001,51.48,51.330002,53.099998,52.25,52.580002,51.299999,51.919998,50.360001,50.330002,50.419998,50.25,50.099998,48.939999,50.400002,49.66,49.700001,49,48.540001,49.459999,47.43,47.59,47.950001,46.400002,47.720001,46.75,47.41,47.459999,47.82,48.310001,50.189999,50.25,49.700001,50.009998,49,48.830002,49.110001,48.389999,48.799999,48.66,48.34,51.43,51.490002,51.849998,51.400002,50.200001,52,52.299999,52.439999,52.400002,51.380001,51.580002,51.700001,51.82,53.650002,51.389999,51.290001,51.669998,50.360001,50.950001,52.130001,52.049999,51.810001,52.689999,53.049999,52.689999,52.59,52.720001,52.450001,51.759998,50.650002,50.490002,50.23,50.259998,50.009998,49.810001,48.400002,47.84,46,46.619999,46.48,47.18,45.490002,45.16,46.5,44.880001,43.889999,43.689999,43.650002,43.84,44.150002,43.889999,43.279999,42.799999,43.32,43.48,45.099998,45.57,46.009998,45.52,46.52,46.52,46.450001,47.889999,48.240002,47.950001,49.689999,49.790001,48.59,48.900002,48.400002,48.32,48.419998,48.099998,46.93,47.860001,47.860001,47.66,47.75,47.709999,45.099998,45.439999,46.59,46.040001,45.32,45.669998,47.240002,48.639999,47.990002,47.91,46.799999,46.709999,46.029999,47.209999,47.299999,47.110001,46.75,48.560001,48.700001,48.650002,48.59,47.200001,48.439999,49.82,49.869999,49.689999,50.110001,49.18,48.130001,47.900002,47.119999,47,47.610001,48.299999]
#MatrizVale = [55.16,56.150002,55.369999,54.529999,54.040001,50.349998,50.119999,49.919998,52.299999,52,52.799999,54.110001,52.880001,52.799999,52.299999,51.290001,50.200001,50.599998,50.369999,50.599998,50.880001,51.25,51.439999,50.009998,49.810001,50.860001,50.439999,49.5,51,51.09,49,52.189999,51.91,52.41,53.689999,53.099998,52.380001,52.599998,52.349998,52.650002,54.23,54.759998,55.279999]
#MatrizBR = [45.004894,48.364414,48.581741,49.6231,49.197502,48.998283,49.414829,49.378605,49.704597,50.555798,49.061672,48.572685,47.685265,46.182083,45.54821,45.104504,44.724178,44.298576,44.733234,44.054089,43.465488,43.067059,43.121387,43.112331,42.514683,42.297356,42.01664,42.01664,42.90406,43.746208,44.008808,43.519821,43.211941,43.184772,43.474541,43.021778,43.628483,43.719036,43.764317,43.646599,42.895004,41.91703,42.487514]
#MatrizPetro = [23.716455,24.377871,24.566847,23.782598,23.85849,23.118547,23.004707,24.218975,24.076677,23.877464,24.152567,24.60792,24.038731,24.247435,23.327248,23.498003,22.236303,22.094006,22.103491,22.160412,21.866327,21.695574,20.870249,21.10741,20.386438,20.443357,21.274069,21.264257,22.255346,23.609507,24.188459,24.257149,24.639847,24.492655,25.00292,24.787039,24.522095,24.384716,24.365091,24.355276,24.688911,24.924417,25.051985]
#RetornosBR = Precos(MatrizBR).Retornos()
#RetornosVale = Precos(MatrizVale).Retornos()
#RetornosPetro = Precos(MatrizPetro).Retornos()
#Pesos = [[0.25,0.25,0.5]]
