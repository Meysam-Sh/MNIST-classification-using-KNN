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
X_train, Y_train = load_date()


"""
This function builds several SVM models with different settings.
Then, it trains each model, applies cross-validation, and computes the average accuracy of each fold.
"""
def SVM(X_train,Y_train):
  SVM1 = SVC(kernel='linear')
  SVM2 = SVC(kernel='poly',degree = 2,gamma = 'scale')
  SVM3 = SVC(kernel='poly',degree = 3,gamma = 'scale')
  SVM4 = SVC(kernel='poly',degree = 10,gamma = 'scale')
  SVM5 = SVC(kernel='rbf',gamma = 'scale')
  SVM6 = SVC(kernel='sigmoid',gamma = 'scale')
  SVM7 = SVC(kernel='precomputed') 
  SVM8 = SVC(kernel='rbf',gamma = 'scale',tol=0.001,decision_function_shape='ovr')
  SVM9 = SVC(kernel='rbf',gamma = 'scale',tol=0.003,decision_function_shape='ovr')
  SVM10 = SVC(kernel='rbf',gamma = 'scale',tol=0.001,decision_function_shape='ovo')
  SVM11 = SVC(kernel='rbf',gamma = 'scale',tol=0.003,decision_function_shape='ovo')

  models = [SVM1,SVM2,SVM3,SVM4,SVM5,SVM5,SVM6,SVM7,SVM8,SVM9,SVM10,SVM11]

  for model in models:
    start  = time.time()
    # model training 
    model.fit(X_train,Y_train)
    # computing the average accuracy of each fold
    test_accuracy = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
    print ('Mean of accuracy is =',round(mean(test_accuracy),2))
    end  = time.time()
    print('elapsed time =',round(end-start,2))
    print('*'*20)

    
