# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  SLR
"""
from env.Include.ml.imports import *
from env.Include.ml.visual import *

def SLR_input(modelName, dataset, config):

  return 0

def SLR_train(modelName, dataset, config):
  funcName = "SLR_train"
  # Select Features
  features_X = [config['x']]
  X = dataset[features_X]
  # Select Target 
  y = dataset[config['y']]

  # Fixed split
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

  # Fitting SLR to the training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(train_X, train_y)

  # Predict
  pred_y = regressor.predict(test_X)
  pred_y = (pred_y > 0.5) 

  show2dScatter(train_X, train_y, config['y'], config['x'], regressor, modelName, config['show'])

  # return regressor, thisModelName, test_y, pred_y

  return {
    'config': config,
    'model': regressor,
    'x' : config['x'], 
    'y' : config['y'],
    'test_y' : test_y,
    'pred_y': pred_y
  }
