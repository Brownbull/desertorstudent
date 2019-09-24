# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  SLR
"""
from env.Include.model.imports_model import *
from env.Include.model.operations import *

def MLR_train(X, Y, config):
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
  Xcols, cols2DropDesc = MLR_optimizeFeatures(X_enc, X_enc_cols, y, thisModelName, config)
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

  return regressor, thisModelName, test_y, pred_y, Xcols, cols2DropDesc

def MLR_optimizeFeatures(X, Xcols, y, thisModelName, config):
  # SET WRITE DIRECTORY
  outDir = "ML_results/" + thisModelName
  if not Path(outDir).exists():
    os.makedirs(outDir)

  # Generate Static columns with one at beggining
  import statsmodels.api as sm
  X["Ones"] = 1
  Xcols = ["Ones"] + Xcols
  
  cols2DropAsc = []

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
            fOpt = open(outDir +"/"+ str(i) + "_Optimization_Summary.txt", 'w+')
            fOpt.write("Columns on Logic:\n")
            fOpt.write("/".join(Xcols) + "\n")
            fOpt.write(str(model_OLS.summary()))
            fOpt.close()
            print(outDir +"/"+ str(len(Xcols)) + "_Optimization_Summary.txt Created")
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
  # Xcols.remove('Ones')
  # Set dropped columns from last to first
  cols2DropDesc = cols2DropAsc[::-1]

  # Write Final Optimization Results
  fOpt = open(outDir +"/"+ str(len(Xcols)) + "_Optimization_Summary.txt", 'w+')
  fOpt.write("Columns on Logic:\n")
  fOpt.write("/".join(Xcols) + "\n")
  fOpt.write(str(model_OLS.summary()))
  fOpt.close()
  print(outDir +"/"+ str(len(Xcols)) + "_Optimization_Summary.txt Created")

  return Xcols, cols2DropDesc