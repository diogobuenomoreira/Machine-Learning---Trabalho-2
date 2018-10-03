import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#-----------------IMPORTANDO DADOS - treino e teste----------------------------

data_train = pd.read_csv('fashion-mnist_train.csv')
data_test = pd.read_csv('fashion-mnist_test.csv')
x = data_train.loc[:,'pixel1':'pixel784']
y = data_train.loc[:,:'label']

#----------------SEPARANDO DADOS EM TREINO, VALIDAÇÃO E TESTE--------------------------

from sklearn.model_selection import train_test_split
x_train, x_validation, y_train, y_validation =  train_test_split(x,y,test_size= 0.2,random_state = 0)
x_test = data_test.loc[:,'pixel1':'pixel784']
y_test = data_test.loc[:,:'label']

#---------------------------Normalização---------------------------------------
x_train = x_train/255.0
x_validation = x_validation/255.0

#------------------------PCA---------------------------------------------------
from sklearn.decomposition import PCA
pca = PCA(n_components = 360)
x_train = pca.fit_transform(x_train)
x_validation = pca.transform(x_validation)
explained_variance = pca.explained_variance_ratio_

#-------------------------------OBTENDO MODELOS--------------------------------

x_train = np.c_[np.ones((len(x_train), 1)), x_train]  # add x0 = 1 para cada coluna
x_validation = np.c_[np.ones((len(x_validation), 1)), x_validation]
#x_test = np.c_[np.ones((len(x_test), 1)), x_test]


# Set the label of the first class to be one, and 0 for othersfrom copy import deepcopy
from copy import deepcopy
y_train = deepcopy(y_train)
idxs_zero = y_train == 0
y_train[idxs_zero] = 1
y_train[-idxs_zero] = 0
y_train = y_train.values

#-Seta Hiperparâmetros-----------------------------------
BATCH_SIZE = 100
STEPS = 5000
LEARNING_RATE = 0.1
N_FEATURES = x_train.shape[1]

def _sigmoid(logits):
    return 1/(1 + np.exp(-logits))

def forward(X, W):
    logits = np.dot(X, W)
    return _sigmoid(logits)[:,0]

def gradient(X, y, pred):
    return np.mean(np.dot((pred - y), X),axis=0)#/#len(y)

def get_next_batch():
  valores = np.random.choice(x_train.shape[0],BATCH_SIZE,replace=False)
  X = []
  Y = []     
  for ind in valores:
      X.append(x_train[ind])
      Y.append(y_train[ind])
  return X, Y

           
# initialize
start = 0
end = BATCH_SIZE
W = np.random.random([N_FEATURES, 1])
custo=[]
passos = []
for step in range(STEPS):

    X_batch, y_batch = get_next_batch()
    pred = forward(X_batch, W)
    dw = gradient(X_batch, y_batch, pred).reshape(N_FEATURES,1) 
    W -= LEARNING_RATE*dw
    
    start += BATCH_SIZE
    end += BATCH_SIZE
    LEARNING_RATE *= .99
    custo.append(np.mean(np.array(pred-y_batch)**2))
    passos.append(step)
    print(custo[-1])

plt.xlabel('Número de Iterações')  
plt.ylabel('Custo')  
plt.title('Custo x Nº Iterações - Regressão Logística')
plt.plot(passos, custo, 'r')
plt.show()


#print('Valores de thetas:', thetas)
#print('Custo final', custo_final)

