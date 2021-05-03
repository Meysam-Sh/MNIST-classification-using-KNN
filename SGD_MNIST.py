# imports
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn import linear_model

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
This function builds several SGD models with different settings.
Then, it trains each model, applies cross-validation, and computes the average accuracy of each fold.
"""
def SGD(X_train, Y_train):
  LC1 = linear_model.SGDClassifier(loss='hinge', penalty='l2', max_iter=10000, tol=1e-3)
  LC1_1 = linear_model.SGDClassifier(loss='hinge', penalty='l1', max_iter=10000, tol=1e-3)
  LC1_2 = linear_model.SGDClassifier(loss='hinge', penalty='elasticnet', max_iter=10000, tol=1e-3)
  LC2 = linear_model.SGDClassifier(loss='log', penalty='l2', max_iter=10000, tol=1e-3)
  LC2_1 = linear_model.SGDClassifier(loss='log', penalty='l1', max_iter=10000, tol=1e-3)
  LC2_2 = linear_model.SGDClassifier(loss='log', penalty='elasticnet', max_iter=10000, tol=1e-3)
  LC3 = linear_model.SGDClassifier(loss='modified_huber', penalty='l2', max_iter=10000, tol=1e-3)
  LC3_1 = linear_model.SGDClassifier(loss='modified_huber', penalty='l1', max_iter=10000, tol=1e-3)
  LC3_2 = linear_model.SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=10000, tol=1e-3)
  LC4 = linear_model.SGDClassifier(loss='squared_hinge', penalty='l2', max_iter=10000, tol=1e-3)
  LC4_1 = linear_model.SGDClassifier(loss='squared_hinge', penalty='l1', max_iter=10000, tol=1e-3)
  LC4_2 = linear_model.SGDClassifier(loss='squared_hinge', penalty='elasticnet', max_iter=10000, tol=1e-3)
  LC5 = linear_model.SGDClassifier(loss='perceptron', penalty='l2', max_iter=10000, tol=1e-3)
  LC5_1 = linear_model.SGDClassifier(loss='perceptron', penalty='l1', max_iter=10000, tol=1e-3)
  LC5_2 = linear_model.SGDClassifier(loss='perceptron', penalty='elasticnet', max_iter=10000, tol=1e-3)

  models = [LC1,LC1_1,LC1_2,LC2,LC2_1,LC2_2,LC3,LC3_1,LC3_2,LC4,LC4_1,LC4_2,LC5,LC5_1,LC5_2]

  for model in models:
    start  = time.time()
    # model training 
    model.fit(X_train,Y_train)
    # computing the average accuracy of each fold
    test_accuracy = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
    print ('Mean of accuracy is =', round(mean(test_accuracy),3))
    end  = time.time()
    print('elapsed time =',round(end-start,3))
    print('*'*20)




    
    
