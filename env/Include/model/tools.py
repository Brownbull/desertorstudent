import os
import pandas as pd
from pathlib import Path
from env.Include.model.imports import *

def checkIfexists(key, tree):
  if (key in tree) and tree[key] is not None:
    return True
  return False

def setOrCreatePath(outDir):
  # SET WRITE DIRECTORY
  if not Path(outDir).exists():
    os.makedirs(outDir)

def OLS_optimizeFeatures(X, Xcols, y, thisModelName, config):
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