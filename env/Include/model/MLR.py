# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  SLR
"""
from env.Include.model.imports import *
from env.Include.model.tools import *
from env.Include.model.processing import *
from env.Include.model.visual import *

def MLR_train(thisModelName, X, Y, config):
  # Model Name
  thisModelName = "MLR_" + config['xColNames'] + "_vs_" + config['y']

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
  Xcols, cols2DropDesc = OLS_optimizeFeatures(X_enc, X_enc_cols, y, thisModelName, config)
  # Xcols, cols2DropDesc = MLR_optimizeFeatures(X, config['x'], y, thisModelName, config)
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
  showCorrHeatMap(df, thisModelName, config['xColNames'], config['y'], config['show'])
  
  return {
    'config': config,
    'model': regressor,
    'x' : Xcols, 
    'y' : config['y'],
    'test_y' : test_y,
    'pred_y': pred_y
  }