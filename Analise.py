#############################################################################
#                            Trabalho de IIA                                #
# Feito por:                                                                #
# - Gabriel Henrique do Nascimento Neres - 221029140                        #
# - Guilherme Nonato da Silva - 221002020                                   #
#############################################################################
# Foi identificado que um dado foi deixado sem a sua label na linha 269
# Este dado foi trocado para ser indicado 


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder  #Possibilita a transformação de classes Alpha-numéricas em classes numéricas.
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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
for inf in dados["Text"]:
    Length.append(len(inf)) # O comprimento de cada mensagem
    letraInicial.append(inf[0].isupper()) # Um boolean indicando se a primeira letra da frase é maiúscula
    pontoFinal.append(inf[-1] in ".?!") # Outro boolean indicando se a mensagem tem pontuação no final
  
# Inserem novas colunas aos dados
dados.insert(1, "Length", Length, True)
dados.insert(2, "Maiusculo?", letraInicial, True)
dados.insert(3, "Ponto Final?", pontoFinal, True)

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
# Foi definido que 30% dos dados serão utilizados para testes

###############################################################################################

# LDA
# Definição do modelo
modelo = LinearDiscriminantAnalysis()
D = modelo.fit_transform(X_train, y_train) # Determina os Descriminantes gerados a partir do modelo  
# Criamos um data frame com os valores do discriminate e as classes relacionadas
plotModelo = pd.DataFrame(D, columns=['Descriminante'])
plotModelo['Classes'] = y_train
sns.FacetGrid(plotModelo, hue ="Classes", height = 6).map(plt.scatter, 'Descriminante', 'Classes')
plt.legend(loc='upper right')
plt.show()  # Mostra a relação do valor do Discriminante gerado e a classe relacionada

modelo.fit(X_train, y_train)    # Utiliza os dados de treinamento parar gerar um modelo preditor
Y_predict = modelo.predict(X_test) # Prediz as possíveis classes usando os dados de teste

# Calcula e mostra a precisão da previsão
precisao = accuracy_score(y_test, Y_predict)
matriz = confusion_matrix(y_test, Y_predict)
print(f'Precisão: {precisao:.3f}')

# Heatmap para representar a Matriz de Confusão, que quantifica os erros e acertos da previsão
plt.figure(figsize=(6, 6))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
plt.xlabel("Predicted")
plt.ylabel("Real Values")
plt.title("Matriz de Confusão")
plt.show()