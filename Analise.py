#############################################################################
#                            Trabalho de IIA                                #
# Feito por:                                                                #
# - Gabriel Henrique do Nascimento Neres - 221029140                        #
# - Guilherme Nonato da Silva - 221002020                                   #
#############################################################################
# Foi identificado que um dado foi deixado sem a sua label na linha 269
# Este dado foi trocado para ser indicado 


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder #Possibilita a transformação de classes Alpha-numéricas em classes numéricas.
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

# Aqui os dados que serão analisados são importados
dados = pd.read_csv("LLM.csv")

# Define o convertor das classes
conversor = LabelEncoder()
y = conversor.fit_transform(dados.iloc[:,-1])

# Para fazer uso dos diferentes classificadores, é preciso determinar parâmetros relevantes para a análise.
# Três parâmetros foram considerados:
Length = []
letraInicial = []
pontoFinal = []
temNumero = []
for inf in dados["Text"]:
    # Variáveis escolhidas inicialmente
    Length.append(len(inf)) # O comprimento de cada mensagem
    letraInicial.append(inf[0].isupper()) # Um boolean indicando se a primeira letra da frase é maiúscula
    pontoFinal.append(inf[-1] in ".?!") # Um boolean indicando se a mensagem tem pontuação no final
    
    # Variável escolhida após erro de colinearidade no QDA
    flag = False
    for i in inf:
        if i.isdigit():
            flag = True
            break
    temNumero.append(flag) # Um boolean indicando a presença de números na mensagem

# Inserem novas colunas aos dados
dados.insert(1, "Length", Length, True)
dados.insert(2, "Maiusculo?", letraInicial, True)
dados.insert(3, "Ponto Final?", pontoFinal, True)
dados.insert(4, "Tem Numero?", temNumero, True)

###############################################################################################

# Para buscar analisar os dados coletados, usamos alguns métodos para facilitar.
# Pair Plot para tentar ver de maneira gráfica como cada variável interage com a outra.
# No nosso caso, serão 9 gráficos.
# ax = sns.pairplot(dados.iloc[:,1:], hue='Label', markers=["o", "s"])
# plt.suptitle("Pair Plot dos dados de texto")
# sns.move_legend(
#     ax, "lower center",
#     bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
# plt.tight_layout()
# plt.show()

# Histograma analisa quantos de cada valor de uma variável é encontrado:
# No nosso caso, são 3 histogramas
# plt.figure(figsize=(12, 6))
# for i, feature in enumerate(["Length", "Maiusculo?", "Ponto Final?"]):
#     plt.subplot(2, 2, i + 1)
#     sns.histplot(data=dados.iloc[:,1:], x=feature, hue='Label', kde=True)
#     plt.title(f'{feature} Distribution')

# Matriz de correlação, busca quantificar a relação entre as variáveis escolhidas, para uma possível exclusão de alguma. 
# correlation_matrix = dados.iloc[:,1:].corr(numeric_only = True)
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title("Correlation Heatmap")
# plt.show()

###############################################################################################

# Pela análise feita, todos as variáveis escolhidas são muito relacionadas, principalmente a de letra maiúscula no começo da mensagem e a presença de pontuações no final. 
# Seria possível remover alguma dessas duas variáveis finais, mantendo resultados finais muito próximos.
# Foi optado por manter todas as variáveis, para possibilitar uma melhor diferenciação nos diferentes métodos que seriam utilizados.

###############################################################################################

# Agora, após ter feito a preparação das variáveis que serão utilizadas, pode ser feita a divisão dos testes e o bloco de treinamento.
X_train, X_test, y_train, y_test = train_test_split(dados.iloc[:, 1:-1], y, test_size=0.3)
X_trainQDA, X_testQDA, y_trainQDA, y_testQDA = train_test_split(dados.iloc[:, [1,4]], y, test_size=0.3)
# Foi definido que 30% dos dados serão utilizados para testes

###############################################################################################

def LDA(X_train, X_test, y_train, y_test, disablePlot:bool = False):
    """ 
        Gera um modelo LDA e retorna a lista de probabilidade dos y dos x de teste.
        Por padrão mostra a respectiva Matriz de Confusão com os dados fornecidos.
    """
    # Definição do modelo
    modeloLDA = LinearDiscriminantAnalysis()
    D = modeloLDA.fit_transform(X_train, y_train) # Utiliza os dados de treinamento parar gerar um modelo preditor e determina os Descriminantes gerados a partir do modelo  
    Y_predict = modeloLDA.predict(X_test) # Prediz as possíveis classes usando os dados de teste

    if (disablePlot == False):
        # Criamos um data frame com os valores do discriminate e as classes relacionadas
        plotModelo = pd.DataFrame(D, columns=['Descriminante'])
        plotModelo['Classes'] = y_train
        sns.FacetGrid(plotModelo, hue ="Classes", height = 6).map(plt.scatter, 'Descriminante', 'Classes')
        plt.legend(loc='upper right')
        plt.show()  # Mostra a relação do valor do Discriminante gerado e a classe relacionada

    # Calcula e mostra a precisão da previsão
    precisao = accuracy_score(y_test, Y_predict)
    print(f'Precisão: {precisao:.3f}')

    if (disablePlot == False):
        # Heatmap para representar a Matriz de Confusão, que quantifica os erros e acertos da previsão
        matriz = confusion_matrix(y_test, Y_predict)
        plt.figure(figsize=(6, 6))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
        plt.xlabel("Predicted")
        plt.ylabel("Real Values")
        plt.title("Matriz de Confusão")
        plt.show()

    # Observamos uma boa precisão utilizando este modelo para predizer os dados de teste.

    return modeloLDA.predict_proba(X_test)[:,1] # Pega apenas as probabilidades, sem a classe

###############################################################################################

def QDA(X_train, X_test, y_train, y_test, disablePlot:bool = False):
    """ 
        Gera um modelo QDA e retorna a lista de probabilidade dos y dos x de teste.
        Por padrão mostra a respectiva Matriz de Confusão com os dados fornecidos.
    """
    # Nos primeiros testes, foi verificado que as variáveis escolhidas são colineares.
    # Esse informação tinha sido obtida pelas análises do Heatmap gerado no início usando apenas os dados iniciais, indicando uma forte relação entre as variáveis escolhidas (mais 0,6).
    # Também foi confirmada pelo código, o qual retorna warnings afirmando que as variáveis utilizadas são colineares.
    # Como este modelo necessita de pelo menos duas variáveis independentes, ele não conseguirá fazer previsões sem a escolha de mais variáveis. Logo, a resposta será sempre a mesma (0). 
    # Para resolver o problema, foi adicionada uma nova variável.

    # Definição do modelo
    modeloQDA = QuadraticDiscriminantAnalysis()
    modeloQDA.fit(X_train, y_train)  # Utiliza os dados de treinamento parar gerar um modelo preditor
    Y_predict = modeloQDA.predict(X_test) # Prediz as possíveis classes usando os dados de teste

    # Calcula e mostra a precisão da previsão
    precisao = accuracy_score(y_test, Y_predict)
    print(f'Precisão: {precisao:.3f}')

    if (disablePlot == False):
        # Heatmap para representar a Matriz de Confusão, que quantifica os erros e acertos da previsão
        matriz = confusion_matrix(y_test, Y_predict)
        plt.figure(figsize=(6, 6))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
        plt.xlabel("Predicted")
        plt.ylabel("Real Values")
        plt.title("Matriz de Confusão")
        plt.show()

    return modeloQDA.predict_proba(X_test)[:,1] # Pega apenas as probabilidades, sem a classe

###############################################################################################

def KNN(X_train, X_test, y_train, y_test, disablePlot:bool = False):
    """ 
        Gera um modelo K-NN e retorna a lista de probabilidade dos y dos x de teste.
        Por padrão mostra a respectiva Matriz de Confusão e o gráfico dos testes com diferentes Ks.
    """
    # Para criar o modelo, precisamos saber um bom número para K.
    # Para definir esse valor, usaremos os valores da precisão de cada valor de k
    Ks = [k for k in range (3,101, 2)] # Os valores de k poderão ser números impares de 3 a 101.
    precisao = [] # Irá armazenar a precisão de cada valor de k
    melhorPrecisao = 0
    melhorK = 1

    # Determina o melhor K
    for k in Ks:
        knn = KNeighborsClassifier(n_neighbors=k)   # Define o modelo K-NN
        knn.fit(X_train, y_train)                   # Treina o modelo
        Y_predict = knn.predict(X_test)             # Gera os valores de Y preditos
        accuracy = accuracy_score(y_test, Y_predict)# Determina a precisão dos valores gerados
        precisao.append(accuracy)                   # Adiciona a lista
        if (accuracy> melhorPrecisao):              # Determina o melhor k
            melhorK = k
            melhorPrecisao = accuracy

    knn = KNeighborsClassifier(n_neighbors= melhorK)
    knn.fit(X_train, y_train)                   # Treina o modelo
   
    print("Melhor K: ", melhorK)
    print(f"Precisão: {melhorPrecisao:.3f}")
    
    if (disablePlot == False):
        # Podemos ver a variação da precisão de cada K no gráfico gerado por:
        sns.lineplot(x = Ks, y = precisao, marker = 'o')
        plt.xlabel("Valores de K")
        plt.ylabel("Precisão")
        plt.show()

        Y_predict = knn.predict(X_test)
        matriz = confusion_matrix(y_test, Y_predict)
        # Heatmap para representar a Matriz de Confusão, que quantifica os erros e acertos da previsão
        plt.figure(figsize=(6, 6))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
        plt.xlabel("Predicted")
        plt.ylabel("Real Values")
        plt.title("Matriz de Confusão")
        plt.show()

    return knn.predict_proba(X_test)[:,1] # Pega apenas as probabilidades, sem a classe

###############################################################################################

def RandomForest(X_train, X_test, y_train, y_test, disablePlot:bool = False): 
    """ 
        Gera um modelo de Random Forest e retorna a lista de probabilidade dos y dos x de teste.
        Por padrão mostra a respectiva Matriz de Confusão.
    """
    # Cria um modelo com 10 árvores de decisão
    RFC = RandomForestClassifier(n_estimators=10, n_jobs=-1, verbose = 1)
    RFC.fit(X_train, y_train)

    # Calcula e mostra a precisão da previsão
    Y_predict = RFC.predict(X_test)
    precisao = accuracy_score(y_test, Y_predict)
    print(f'Precisão: {precisao:.3f}')

    if (disablePlot == False):
        matriz = confusion_matrix(y_test, Y_predict)
        # Heatmap para representar a Matriz de Confusão, que quantifica os erros e acertos da previsão
        plt.figure(figsize=(6, 6))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
        plt.xlabel("Predicted")
        plt.ylabel("Real Values")
        plt.title("Matriz de Confusão")
        plt.show()

    return RFC.predict_proba(X_test)[:,1] # Pega apenas as probabilidades, sem a classe

###############################################################################################

def SVM(X_train, X_test, y_train, y_test, disablePlot:bool = False):
    """
        Gera um modelo de SVM e retorna a lista de probabilidade dos y dos x de teste.
        Por padrão mostra a respectiva Matriz de Confusão.
    """
    # Cria o modelo
    svm = SVC(probability=True) # Sem definir o tipo de kernel a ser usado 
    svm.fit(X_train, y_train)

    # Calcula e mostra a precisão da previsão
    Y_predict = svm.predict(X_test)
    precisao = accuracy_score(y_test, Y_predict)
    print(f'Precisão: {precisao:.3f}')

    if (disablePlot == False):
        matriz = confusion_matrix(y_test, Y_predict)
        # Heatmap para representar a Matriz de Confusão, que quantifica os erros e acertos da previsão
        plt.figure(figsize=(6, 6))
        sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
        plt.xlabel("Predicted")
        plt.ylabel("Real Values")
        plt.title("Matriz de Confusão")
        plt.show()

    return svm.predict_proba(X_test)[:,1] # Pega apenas as probabilidades, sem a classe

###############################################################################################
# Remova o parâmetro True ao final para ver os gráficos internos
LDA_Proba = LDA(X_train, X_test, y_train, y_test, True)
QDA_Proba = QDA(X_trainQDA, X_testQDA, y_trainQDA, y_testQDA, True)
KNN_Proba = KNN(X_train, X_test, y_train, y_test, True)
RFC_Proba = RandomForest(X_train, X_test, y_train, y_test, True)
SVM_Proba = SVM(X_train, X_test, y_train, y_test, True)

# Curvas ROC e valores AUC
def ROC(true_y, y_prob, nome):
    """ Função utilizada para plotar a curva ROC """

    # Se tiver NaN nos valores, substitui por 0, como é tratado pelos preditores
    if (True in np.isnan(y_prob)):
        for index, i in enumerate(np.isnan(y_prob)):
            if i == True:
                y_prob[index] = 0

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print(nome + " : " + str(roc_auc_score(true_y, y_prob).round(3)))

ROC(y_test, LDA_Proba, "AUC LDA")
ROC(y_testQDA, QDA_Proba, "AUC QDA")
ROC(y_test, KNN_Proba, "AUC K-NN")
ROC(y_test, RFC_Proba, "AUC Random Forest")
ROC(y_test, SVM_Proba, "AUC SVM")
plt.legend(["LDA","QDA","KNN", "Random Forest", "SVM"], loc="lower right")
plt.show()

# Foi difícil determinar os métodos que apresentam os melhores resultados, 
# levando em consideração as variáveis escolhidas e os resultados observados, 
# além de que os valores de AUC da maioria são muito próximos.
#
# Ao executar diversas vezes, observou-se uma tendência entre os modelos para apresentar um melhor resultado. 
# A conclusão obtida foi:
# LDA               AUC normalmente próximo de 1
# Random Forest     AUC normalmente próximo de 0,997
# K-NN              AUC normalmente próximo de 0,995
# SVM               AUC normalmente próximo de 0,910
# QDA               AUC normalmente próximo de 0,870