# imports
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.ensemble import RandomForestClassifier

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
This function builds several random forest models with different settings.
Then, it trains each model, applies cross-validation, and computes the average accuracy of each fold.
"""
def RF(X_train, Y_train):
  RF1 = RandomForestClassifier(max_features='auto', n_estimators=10)
  RF2 = RandomForestClassifier(max_features='sqrt', n_estimators=10)
  RF3 = RandomForestClassifier(max_features='log2', n_estimators=10)
  RF4 = RandomForestClassifier(max_features='auto', n_estimators=100)
  RF5 = RandomForestClassifier(max_features='log2', n_estimators=100)
  RF6 = RandomForestClassifier(max_features='auto', n_estimators=500)
  RF7 = RandomForestClassifier(max_features='log2', n_estimators=500)

  models = [RF1, RF2, RF3, RF4, RF5, RF6, RF7]

  for model in models:
    start  = time.time()
    # model training 
    model.fit(X_train,Y_train)
    # computing the average accuracy of each fold
    test_accuracy = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
    print ('Mean of accuracy is =',round(mean(test_accuracy),3))
    end  = time.time()
    print('elapsed time =',round(end-start,2))
    print('*'*20)
    
