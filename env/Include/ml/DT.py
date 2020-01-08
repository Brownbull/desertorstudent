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
    X_enc_cols, cols2DropDesc = backwardElimination(X_enc, X_enc_cols, y, folderPath, modelName, config)
     # Set Optimal cols
    X_enc = X_enc[X_enc_cols]

  # Fixed split
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size = 0.2, random_state = 0)

  # Feature Scaling
  train_X, test_X = fScaling(train_X, test_X)

  return {
    'Xcols': X_enc_cols,
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
        # , label="none"
        # ,proportion=True
        )
    from subprocess import check_output
    check_output("dot -Tpng " + outDir + "/tree.dot > " + outDir + "/tree.png", shell=True)
  
  # VISUALIZATION
  df = ds['X_enc']
  df[config['y']] = ds['y']
  showCorrHeatMap(df, modelName, config['xColNames'], config['y'], config['show'])

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

  # return classifier, modelName, test_y, pred_y, Xcols, X_enc
  return {
    'config': config,
    'model': classifier,
    'x' : ds['Xcols'], 
    'y' : config['y'],
    'test_x' : ds['test_X'],
    'test_y' : ds['test_y'],
    'pred_y_raw': pred_y_raw,
    'pred_y': pred_y
  }