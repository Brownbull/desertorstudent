# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  SLR
"""
from env.Include.ml.imports import *
from env.Include.ml.visual import *

import xlwt
from xlwt.Workbook import *
from pandas import ExcelWriter
import xlsxwriter

def SLR_input(folderPath, modelName, dataset, config):
  funcName = "SLR_input"
  # Select Features
  features_X = [config['x']]
  X = dataset[features_X]
  # Select Target 
  y = dataset[config['y']]

  # Fixed split
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)

  # Check log and proceed
  if checkIfexists('log', config):
    if config['log'] == 'all':
      # Save train and test data to excel
      excelJson = [
        {
          "sheetName": 'train',
          "sheetData": [ train_X, train_y ]
        },
        {
          "sheetName": 'test',
          "sheetData": [ test_X, test_y ]
        }
      ]
      saveDFs2xlsx(folderPath, funcName, excelJson, False, "df")

  return {
    'train_X': train_X,
    'test_X': test_X,
    'train_y': train_y,
    'test_y': test_y
  }

def SLR_train(folderPath, modelName, ds, config):
  funcName = "SLR_train"

  # Fitting SLR to the training set
  from sklearn.linear_model import LinearRegression
  regressor = LinearRegression()
  regressor.fit(ds['train_X'], ds['train_y'])

  # Predict
  pred_y_raw = regressor.predict(ds['test_X'])
  pred_y = (pred_y_raw > 0.5) 

  # Visualization
  show2dScatter(ds['train_X'], ds['train_y'], config['y'], config['x'], regressor, modelName, config['show'])

  return {
    'config': config,
    'model': regressor,
    'x' : config['x'], 
    'y' : config['y'],
    'test_y' : ds['test_y'],
    'pred_y_raw': pred_y_raw,
    'pred_y': pred_y
  }
