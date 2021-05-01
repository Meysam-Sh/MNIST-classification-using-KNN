# imports
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.svm import SVC

# load MNIST dataset 
def load_date():
  mnist = tf.keras.datasets.mnist
  (train_data,train_labels),(test_data,test_labels) = mnist.load_data()

  # Flatten the train_data and test_data for faster computation
  train_data = train_data.reshape([60000,784])
  test_data = test_data.reshape([10000,784])

  # convertting from int8 to float32  
  train_data = np.float32(train_data)
  test_data = np.float32(test_data)
  X_train = train_data[0:30000,:]
  Y_train = train_labels[0:30000]
  return X_train, Y_train

# data point features and lables 
X_train, Y_train = load_date()ß


"""
This function builds several MLP models with different settings.
Then, it trains each model, applies cross-validation, and computes the average accuracy of each fold.
"""
def MPL(X_train, Y_train):
  MLP1 = MLPClassifier(hidden_layer_sizes=(200),activation='relu', max_iter=10000)  
  MLP2 = MLPClassifier(hidden_layer_sizes=(400,400),activation='relu', max_iter=10000) 
  MLP3 = MLPClassifier(hidden_layer_sizes=(800, 800,800),activation='relu', max_iter=10000)
  MLP4 = MLPClassifier(hidden_layer_sizes=(200),activation='identity', max_iter=10000)
  MLP5 = MLPClassifier(hidden_layer_sizes=(400, 400),activation='identity', max_iter=10000)
  MLP6 = MLPClassifier(hidden_layer_sizes=(800, 800),activation='identity', max_iter=10000)
  MLP7 = MLPClassifier(hidden_layer_sizes=(200),activation='logistic', max_iter=10000)
  MLP8 = MLPClassifier(hidden_layer_sizes=(400, 400),activation='logistic', max_iter=10000)
  MLP9 = MLPClassifier(hidden_layer_sizes=(800, 800, 800),activation='logistic', max_iter=10000)
  MLP10 = MLPClassifier(hidden_layer_sizes=(200),activation='tanh', max_iter=10000)
  MLP11 = MLPClassifier(hidden_layer_sizes=(400, 400),activation='tanh', max_iter=10000)
  MLP12 = MLPClassifier(hidden_layer_sizes=(800, 800, 800),activation='tanh', max_iter=10000)

  models = [MLP1,MLP2,MLP3,MLP4,MLP5,MLP6,MLP7,MLP8,MLP9,MLP10,MLP11,MLP12]

  for model in models:
    start  = time.time()
    # model training 
    model.fit(X_train,Y_train)
    # computing the average accuracy of each fold
    test_accuracy = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
    print ('cross-validation =',round(mean(test_accuracy),2))
    end  = time.time()
    print('elapsed time =',round(end-start,2))
    print('*'*20)    

    
