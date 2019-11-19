import numpy as np
import scipy.stats.stats as sss

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
        print(MatrizRetornos)
        RetornosTransp = Utilidades(MatrizRetornos).Transpose()
        return(np.matmul(Pesos,[RetornosTransp]))
    
def PortfolioAleatorio(self, N):
        RetornosAtivos = self.RetornosAtivos
        NAtivos = len(RetornosAtivos)
        MatrizResposta = []
        FO = open(" - - - - - - - - - Desktop/TesteMarkowitzAleat.txt",'w+')
        for x in range(N):
            Pesos = Utilidades().PesosAleatorios(NAtivos)
            ObjMarkowitz = Markowitz(RetornosAtivos, [Pesos])
            VarPort = ObjMarkowitz.Variancia()
            VolPort = np.sqrt(VarPort)
            RetPort = ObjMarkowitz.ERetorno()
            LinhaPesos = []
            LinhaRisco = []
            strRisco = str(VolPort[0]) + "|" + str(RetPort[0])
            for w in range(NAtivos+2):
                #print(Pesos[k] for k in range(len(Pesos)))
                LinhaPesos = [Pesos[k] for k in range(len(Pesos))] #+ str(VolPort[0]) + str(RetPort[0]))
                #for k in range(len(Pesos)):
                    #LinhaPesos[k] = str(LinhaPesos[k]) + strRisco
            #LinhaRisco = [str(VolPort[0]) + str(RetPort[0])]
            #print("Risco " + str(VolPort[0]) + "|" + str(RetPort[0]))
            MatrizResposta.append(LinhaPesos)
        for y in range(len(MatrizResposta)):
            FO.write(str(MatrizResposta[y]))
            FO.write('\n')
        FO.close()
        return(MatrizResposta)
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
