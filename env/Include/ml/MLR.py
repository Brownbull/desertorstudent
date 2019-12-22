# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  MLR
"""
from env.Include.ml.imports import *
from env.Include.ml.processing import *
from env.Include.ml.visual import *
from env.Include.ml.optimize import *

def MLR_train(modelName, X, Y, config):
  # Select Features
  X = X[config['x']]
  # Select Target 
  y = Y
  
  # ENCODE DATA
  if config['xCategorical']:
    X_bin = cat2Number(X, config['xCategorical'])
    X_enc = cat2Dummy(X_bin, config['xCategorical'])
  else:
    X_enc = X
  X_enc_cols = list(X_enc.columns.values)

  # MLR Optimize
  Xcols, cols2DropDesc = backwardElimination(X_enc, X_enc_cols, y, modelName, config)
  # Xcols, cols2DropDesc = MLR_optimizeFeatures(X, config['x'], y, modelName, config)
  # Set Optimal cols
  X_enc = X_enc[Xcols]

  # Fixed split
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size = 0.2, random_state = 0)

  # Fitting SLR to the training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(train_X, train_y)

  # Predict
  pred_y = regressor.predict(test_X)
  pred_y = (pred_y > 0.5)
  
  regressor.fit(train_X, train_y)

  # Show graph
  df = X_enc
  df[config['y']] = y
  showCorrHeatMap(df, modelName, config['xColNames'], config['y'], config['show'])
  
  return {
    'config': config,
    'model': regressor,
    'x' : Xcols, 
    'y' : config['y'],
    'test_y' : test_y,
    'pred_y': pred_y
  }