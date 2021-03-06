# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  MLR
"""
from env.Include.ml.imports import *
from env.Include.ml.processing import *
from env.Include.ml.visual import *
from env.Include.ml.optimize import *

def MLR_input(folderPath, modelName, X, Y, config):
  funcName = "MLR_input"
  # Select Features
  X = X[config['x']]
  # Select Target 
  y = Y
  cols2DropDesc = []
  explained_variance = 0
  
  # ENCODE DATA
  if config['xCategorical']:
    X_bin = cat2Number(X, config['xCategorical'])
    X_enc = cat2Dummy(X_bin, config['xCategorical'])
  else:
    X_enc = X
  X_enc_cols = list(X_enc.columns.values)

  # Optimize BackwardElimination
  if "BWE" in map(str.upper, config["Optimize"]):
    X_enc_cols, cols2DropDesc = backwardElimination(X_enc, X_enc_cols, y, folderPath, modelName, config)
    # Set Optimal cols
    X_enc = X_enc[X_enc_cols]

  # Fixed split
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size = 0.2, random_state = 0)

  # Feature Scaling
  train_X, test_X = fScaling(train_X, test_X)

  # Optimize PrincipalComponentAnalysis
  if "PCA" in map(str.upper, config["Optimize"]):
    train_X, test_X, explained_variance = principalComponentAnalysis(folderPath, modelName, config, train_X, test_X)
  
  return {
    'Xcols': X_enc_cols,
    'cols2DropDesc': cols2DropDesc,
    'X_enc': X_enc,
    'y': y,
    'train_X': train_X,
    'test_X': test_X,
    'train_y': train_y,
    'test_y': test_y,
    'explained_variance' : explained_variance
  } 
  

def MLR_train(folderPath, modelName, ds, config):
  funcName = "MLR_train"
  # Fitting SLR to the training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(ds['train_X'], ds['train_y'])

  # Predict
  pred_y_raw = regressor.predict(ds['test_X'])
  pred_y = (pred_y_raw > 0.5) 

  # DATA SAVE
  ds['train_X'] = pd.DataFrame(ds['train_X']) 
  ds['test_X'] = pd.DataFrame(ds['test_X']) 
  if type(config["Optimize"]) != type(True):
    if "PCA" not in map(str.upper, config["Optimize"]):
      ds['train_X'].columns = ds['Xcols']
      ds['test_X'].columns = ds['Xcols']
  else:
    ds['train_X'].columns = ds['Xcols']
    ds['test_X'].columns = ds['Xcols']

  ds['train_y'] = pd.DataFrame(ds['train_y']) 
  ds['train_y'].columns = ['train_y']
  ds['test_y'] = pd.DataFrame(ds['test_y']) 
  ds['test_y'].columns = ['train_y']
  ds['pred_y_raw'] = pd.DataFrame(pred_y_raw) 
  ds['pred_y_raw'].columns = ['pred_y_raw']
  ds['pred_y'] = pd.DataFrame(pred_y) 
  ds['pred_y'].columns = ['pred_y']
  # Remove column
  if checkIfexists(config['y'], ds['test_X']):
    ds['test_X'].drop(config['y'], axis=1, inplace=True)
  if checkIfexists('Ones', ds['test_X']):
    ds['test_X'].drop('Ones', axis=1, inplace=True)
  
  # VISUALIZATION
  df = ds['test_X']
  df[config['y']] = ds['test_y']
  showCorrHeatMap(df, modelName, config['xColNames'], config['y'], config['show'])

  excelJson = [
    {
      "sheetName": 'train',
      "sheetData": [ ds['train_X'], ds['train_y'] ]
    },
    {
      "sheetName": 'test',
      "sheetData": [ ds['test_X'], ds['test_y'], ds['pred_y_raw'], ds['pred_y']  ]
    }
  ]
  save2xlsx(folderPath, funcName, excelJson, False, "df")
  
  return {
    'config': config,
    'model': regressor,
    'x' : ds['Xcols'], 
    'y' : config['y'],
    'test_x' : ds['test_X'],
    'test_y' : ds['test_y'],
    'pred_y_raw': pred_y_raw,
    'pred_y': pred_y
  }