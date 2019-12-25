# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  Decision Tree
"""
from env.Include.ml.imports import *
from env.Include.ml.processing import *
from env.Include.ml.visual import *
from env.Include.ml.optimize import *

def DT_input(folderPath, modelName, X, Y, config):
  funcName = "DT_input"
  # Select Features
  Xcols = config['x']
  X = X[Xcols]
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
  if config['Optimize']:
    Xcols, cols2DropDesc = backwardElimination(X_enc, X_enc_cols, y, folderPath, modelName, config)
    X_enc = X_enc[Xcols]

  # Fixed split
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size = 0.2, random_state = 0)

  # Feature Scaling
  train_X, test_X = fScaling(train_X, test_X)

  return {
    'Xcols': Xcols,
    'cols2DropDesc': cols2DropDesc,
    'X_enc': X_enc,
    'y': y,
    'train_X': train_X,
    'test_X': test_X,
    'train_y': train_y,
    'test_y': test_y
  } 

def DT_train(folderPath, modelName, ds, config):
  funcName = "DT_train"
  # Fitting SLR to the training set
  from sklearn import tree, model_selection
  # Classifier
  classifier = tree.DecisionTreeClassifier(
      random_state=0, max_depth=7, min_samples_split=2)
  classifier = classifier.fit(ds['train_X'], ds['train_y'])

  # Predict
  pred_y_raw = classifier.predict(ds['test_X'])
  pred_y = (pred_y_raw > 0.5) 

  # SHOW GRAPH
  if config['show'] in ['inline', 'file']:
    # SET WRITE DIRECTORY
    outDir = "results/ML/" + modelName
    setOrCreatePath(outDir)

    tree.export_graphviz(
        classifier, feature_names=ds['Xcols'], out_file=outDir + "/tree.dot", 
        filled=True, rounded=True,
        special_characters=True, impurity=False, class_names=True
        # ,proportion=True
        )
    from subprocess import check_output
    check_output("dot -Tpng " + outDir + "/tree.dot > " + outDir + "/tree.png", shell=True)
  
  df = ds['X_enc']
  df[config['y']] = ds['y']
  showCorrHeatMap(df, modelName, config['xColNames'], config['y'], config['show'])

  # return classifier, modelName, test_y, pred_y, Xcols, X_enc
  return {
    'config': config,
    'model': classifier,
    'x' : ds['Xcols'], 
    'y' : config['y'],
    'test_y' : ds['test_y'],
    'pred_y_raw': pred_y_raw,
    'pred_y': pred_y
  }