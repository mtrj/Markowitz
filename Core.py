import numpy as np
import scipy.stats.stats as sss
import matplotlib.pyplot as plt

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
        NAtivos = len(RetornosAtivos)
        MatrizCorrel = []
        for i in range(NAtivos):
            Linha = []
            for x in range(NAtivos):
                Linha.append(sss.pearsonr(RetornosAtivos[i],RetornosAtivos[x])[0])
            #print(Linha)
            MatrizCorrel.append(Linha)
        return(MatrizCorrel)

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
        MatrizResposta = []
        FO = open("C:/Users/mterocha/Desktop/TesteMarkowitzAleat.txt",'w+')
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
            LinhaPesos = [Pesos[k] for k in range(len(Pesos))]
            LinhaRisco = [str(VolPort[0]) + '|' + str(RetPort[0])]
            MatrizResposta.append(str(LinhaPesos) + '|' + str(LinhaRisco))
        for y in range(len(MatrizResposta)):
            FO.write(str(MatrizResposta[y]))
            FO.write('\n')
        FO.close()
        SharpeMax = max(SharpeArr)
        SharpeMaxLoc = SharpeArr.index(SharpeMax)
        VolSM = VolArr[SharpeMaxLoc]
        RetSM = RetArr[SharpeMaxLoc]
        plt.style.use('seaborn-whitegrid')
        plt.xlabel('Volatilidade')
        plt.ylabel('Retorno')
        plt.scatter(VolArr, RetArr, c=SharpeArr, cmap='viridis')
        plt.scatter(VolSM, RetSM,c='red', s=50)
        plt.colorbar(label='Sharpe')
        plt.savefig("C:/Users/mterocha/Desktop/MarkowitzTeste.png")
        #return(MatrizResposta)
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
    
#MatrizVale = [55.16,56.150002,55.369999,54.529999,54.040001,50.349998,50.119999,49.919998,52.299999,52,52.799999,54.110001,52.880001,52.799999,52.299999,51.290001,50.200001,50.599998,50.369999,50.599998,50.880001,51.25,51.439999,50.009998,49.810001,50.860001,50.439999,49.5,51,51.09,49,52.189999,51.91,52.41,53.689999,53.099998,52.380001,52.599998,52.349998,52.650002,54.23,54.759998,55.279999]
#MatrizBR = [45.004894,48.364414,48.581741,49.6231,49.197502,48.998283,49.414829,49.378605,49.704597,50.555798,49.061672,48.572685,47.685265,46.182083,45.54821,45.104504,44.724178,44.298576,44.733234,44.054089,43.465488,43.067059,43.121387,43.112331,42.514683,42.297356,42.01664,42.01664,42.90406,43.746208,44.008808,43.519821,43.211941,43.184772,43.474541,43.021778,43.628483,43.719036,43.764317,43.646599,42.895004,41.91703,42.487514]
#MatrizPetro = [23.716455,24.377871,24.566847,23.782598,23.85849,23.118547,23.004707,24.218975,24.076677,23.877464,24.152567,24.60792,24.038731,24.247435,23.327248,23.498003,22.236303,22.094006,22.103491,22.160412,21.866327,21.695574,20.870249,21.10741,20.386438,20.443357,21.274069,21.264257,22.255346,23.609507,24.188459,24.257149,24.639847,24.492655,25.00292,24.787039,24.522095,24.384716,24.365091,24.355276,24.688911,24.924417,25.051985]
#RetornosBR = Precos(MatrizBR).Retornos()
#RetornosVale = Precos(MatrizVale).Retornos()
#RetornosPetro = Precos(MatrizPetro).Retornos()
#Pesos = [[0.25,0.25,0.5]]
