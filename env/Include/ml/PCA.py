# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  Principal Component Analysis
"""
from env.Include.ml.imports import *
from env.Include.ml.processing import *
from env.Include.ml.visual import *
from env.Include.ml.optimize import *

def PCA_train(modelName, X, Y, config):
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
    Xcols, cols2DropDesc = backwardElimination(X_enc, X_enc_cols, y, modelName, config)
    X_enc = X_enc[Xcols]

  # Fixed split
  from sklearn.model_selection import train_test_split
  train_X, test_X, train_y, test_y = train_test_split(X_enc, y, test_size = 0.2, random_state = 0)

  # Feature Scaling
  train_X, test_X = fScaling(train_X, test_X)

  # Fitting SLR to the training set
  from sklearn import tree, model_selection
  # Classifier
  classifier = tree.DecisionTreeClassifier(
      random_state=0, max_depth=7, min_samples_split=2)
  classifier = classifier.fit(train_X, train_y)

  # Predict
  pred_y = classifier.predict(test_X)
  pred_y = (pred_y > 0.5) 

  # SHOW GRAPH
  if config['show'] in ['inline', 'file']:
    # SET WRITE DIRECTORY
    outDir = "results/ML/" + modelName
    if not Path(outDir).exists():
      os.makedirs(outDir)
    tree.export_graphviz(
        classifier, feature_names=Xcols, out_file=outDir + "/tree.dot", 
        filled=True, rounded=True,
        special_characters=True, impurity=False, class_names=True
        # ,proportion=True
        )
    from subprocess import check_output
    check_output("dot -Tpng " + outDir + "/tree.dot > " + outDir + "/tree.png", shell=True)

  # return classifier, modelName, test_y, pred_y, Xcols, X_enc
  return {
    'config': config,
    'model': classifier,
    'x' : config['x'], 
    'y' : config['y'],
    'test_y' : test_y,
    'pred_y': pred_y
  }