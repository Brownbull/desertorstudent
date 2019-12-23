# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  Optimize Models Features
"""
from env.Include.lib.functions import *
from env.Include.ml.imports import *

def backwardElimination(X, Xcols, y, folderPath, thisModelName, config):
  funcName = "backwardElimination"
  # SET WRITE DIRECTORY
  outDir = "results/ML/" + thisModelName
  if not Path(outDir).exists():
    os.makedirs(outDir)

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
  # Xcols.remove('Ones')
  # Set dropped columns from last to first
  cols2DropDesc = cols2DropAsc[::-1]

  # Write Final Optimization Results
  # fOpt = open(outDir +"/"+ str(len(Xcols)) + "_Optimization_Summary.txt", 'w+')
  # fOpt.write("Columns on Logic:\n")
  # fOpt.write("/".join(Xcols) + "\n")
  # fOpt.write(str(model_OLS.summary()))
  # fOpt.close()
  # print(outDir +"/"+ str(len(Xcols)) + "_Optimization_Summary.txt Created")

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