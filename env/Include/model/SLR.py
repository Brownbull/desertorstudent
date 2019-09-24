# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  SLR
"""
from env.Include.model.imports_model import *
from env.Include.model.operations import *

def SLR_train(dataset, config):
  # Model Name
  thisModelName = "SLR_" + config['x'] + "_vs_" + config['y']

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

  show2dScatter(train_X, train_y, config['y'], config['x'], regressor, thisModelName, config['show'])

  return regressor, thisModelName, test_y, pred_y
