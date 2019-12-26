# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  Optimize Models Features
"""
from env.Include.lib.functions import *
from env.Include.ml.imports import *
from sklearn.decomposition import PCA

def getNComponents(train_X, test_X):
  explained_variance_acc = 0
  n_components = 0
  # Get Optimal Number of components
  pca = PCA(n_components = None, whiten = True)
  train_X = pca.fit_transform(train_X)
  test_X = pca.transform(test_X)
  explained_variance = pca.explained_variance_ratio_
  # Get Components number and total explained variance 
  for i, ev in enumerate(explained_variance):
    explained_variance_acc = explained_variance_acc + ev
    if (explained_variance_acc > 0.75):
      n_components = i + 1
      break
  return n_components, explained_variance_acc

def principalComponentAnalysis(folderPath, modelName, config, train_X, test_X):
  funcName = "principalComponentAnalysis"
  print(modelName + ": Optimizing with " + funcName)

  # Get Optimal numbers of componenets
  n_components, explained_variance_acc = getNComponents(train_X, test_X)

  # Applying PCA
  pca = PCA(n_components = n_components, whiten = True)
  train_X = pca.fit_transform(train_X)
  test_X = pca.transform(test_X)
  explained_variance = pca.explained_variance_ratio_

  # Append Json Element
  covsIdx = ['PCA Variable']
  covsVar = ["Explained Variance Ratio"]
  for i, cv in enumerate(explained_variance):
    covsIdx.append(i + 1) 
    covsVar.append(cv)
  excelJson = [
    {
      "sheetName": "evr",
      "sheetData": [ 
        covsIdx,
        covsVar,
        ["Accumulated variance", explained_variance_acc]
      ]
    } 
  ]
  # Write Excel Optimization File 
  save2xlsx(folderPath, funcName, excelJson, False, "cols")


  train_X = pd.DataFrame(train_X)
  test_X = pd.DataFrame(test_X)

  return train_X, test_X, explained_variance


def backwardElimination(X, Xcols, y, folderPath, modelName, config):
  funcName = "backwardElimination"
  print(modelName + ": Optimizing with " + funcName)

  # Generate Static columns with one at beggining
  import statsmodels.api as sm
  X["Ones"] = 1
  Xcols = ["Ones"] + Xcols
  
  cols2DropAsc = []

  excelJson = []
  # Backward Elimination
  for i in range(0, int(len(Xcols))):
    opt_X = X[Xcols]
    model_OLS = sm.OLS(endog = y, exog = opt_X).fit()
    Pmax = max(model_OLS.pvalues)
    adjR_before = model_OLS.rsquared_adj
    if Pmax > config['SL']:
      for j, col in enumerate(Xcols):
        if model_OLS.pvalues[j] == Pmax:
          cols2DropAsc.append(col)
          # Generate separated copy of current features
          Xcols_Temp = Xcols[:]
          # Remove identified non related feature
          Xcols.remove(col)
          
          # traceOpt
          if config['traceOptimization']:
            # Write Optimization Step
            for js in excelJson:
              if js["sheetName"] == str(len(Xcols)):
                excelJson.remove(js)   
            # Arrange Feature Array  
            colsIn = []
            colsIn.append("Columns on Logic")
            for col in Xcols:
              colsIn.append(col)
            
            # Append Json Element
            excelJson.append( 
              {
                "sheetName": str(len(Xcols)),
                "sheetData": [ 
                  colsIn,
                  ["Summary", str(model_OLS.summary())]
                ]
              } 
            )
          else:
            print("Drop Feature: ", j,": ", col, "- Pval: ", Pmax)
          
          # Checking Rsquared
          temp_opt_X = X[Xcols]
          tmp_regressor = sm.OLS(endog = y, exog = temp_opt_X).fit()
          adjR_after = tmp_regressor.rsquared_adj
          if (adjR_before >= adjR_after):
            # Rollback: no more gain on this point
            opt_X = Xcols_Temp[:]
            break
  
  # Remove ones column
  Xcols.remove('Ones')
  # Set dropped columns from last to first
  cols2DropDesc = cols2DropAsc[::-1]

  # Write Final Optimization Results
  for js in excelJson:
    if js["sheetName"] == str(len(Xcols)):
      excelJson.remove(js)   
  # Arrange Feature Array  
  colsIn = []
  colsIn.append("Columns on Logic")
  for col in Xcols:
    colsIn.append(col)
  
  # Append Json Element
  excelJson.append( 
    {
      "sheetName": str(len(Xcols)),
      "sheetData": [ 
        colsIn,
        ["Summary", str(model_OLS.summary())]
      ]
    } 
  )

  # Write Excel Optimization File 
  save2xlsx(folderPath, funcName, excelJson, False, "rows")

  return Xcols, cols2DropDesc