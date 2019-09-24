import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from env.Include.model.imports_model import *

def checkIfexists(key, tree):
  if (key in tree) and tree[key] is not None:
    return True
  return False

def cat2Number(df, cat_cols):
  label = LabelEncoder()
  # transform words to numbers
  for col in cat_cols:
    df[col] = label.fit_transform(df[col]) 

  return df
  

def cat2Dummy(df, cat_cols):
  # Encode Categorical Data
  for col in cat_cols:
    data1_dummy = pd.get_dummies(df[[col]], columns=[col])
    # Avoid Dummy variable trap
    dummyCols = data1_dummy.columns.tolist()[1:]
    df[dummyCols] = data1_dummy[dummyCols]
    df = df.drop(col, axis=1)

  return df

def evaluateRegModel(test_y, pred_y, thisModelName, modelResults):
  # SET WRITE DIRECTORY
  outDir = "ML_results/" + thisModelName
  if not Path(outDir).exists():
    os.makedirs(outDir)

  # Making the Confusion Matrix
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(test_y, pred_y)

  # Accuracy Values
  # Defs
  TP = cm[0][0]
  TN = cm[1][1]
  FP = cm[0][1]
  FN = cm[1][0]
  # Formulas
  Accuracy = (TP + TN) / (TP + TN + FP + FN) # 70 80 90 Good
  Precision = TP / (TP + FP)
  Recall = TP / (TP + FN)
  F1_Score = 2 * Precision * Recall / (Precision + Recall)

  # Write Evaluation Results
  fRes = open(outDir +"/Evaluation.txt", 'w+')
  fRes.write('{:>5} \n'.format('Model:' + thisModelName))
  fRes.write('{:>10} {:>10}\n'.format('x:', str(", ".join(modelResults['x']))))
  # fRes.write('{:>10} {:>10}\n'.format('x:', modelResults['config']['xColNames'] if checkIfexists('xColNames', modelResults['config']) else modelResults['x'] ))
  fRes.write('{:>10} {:>10}\n'.format('y:', modelResults['y']))
  fRes.write('{:>15}  {:>20}\n'.format('Accuracy:', Accuracy))
  fRes.write('{:>15}  {:>20}\n'.format('Precision:', Precision))
  fRes.write('{:>15}  {:>20}\n'.format('Recall:', Recall))
  fRes.write('{:>15}  {:>20}\n'.format('F1_Score:', F1_Score))
  fRes.close()
  print(outDir +"/Evaluation.txt Created")

def show2dScatter(train_X, train_y, y, x, regressor, thisModelName, show):
  if show in ['inline', 'file']:
    graphCongInit()
    # Visualising 
    plt.scatter(train_X, train_y, color='red')
    plt.plot(train_X, regressor.predict(train_X), color='blue')
    plt.title(str("'{0}' vs '{1}' SLR prediction (training set)".format(y, x)))
    plt.xlabel("{0}".format(x))
    plt.ylabel("{0}".format(y))
    if show == 'inline':
      plt.show() 
    elif show == 'file':
      # SET WRITE DIRECTORY
      outDir = "ML_results/" + thisModelName
      if not Path(outDir).exists():
        os.makedirs(outDir)
      plt.savefig(outDir + "/" + x + "_vs_" + y + '.png', bbox_inches='tight')
      print(outDir + "/" + x + "_vs_" + y + ".png Created")
  else:
    print("Missconfigured show")

def showCorrHeatMap(df, thisModelName, x, y, show):
  if show in ['inline', 'file']:
    _ , ax = plt.subplots(figsize =(20, 16))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
  plt.title('Pearson Correlation of Features', y=1.05, size=15)
  if show == 'inline':
      plt.show() 
  elif show == 'file':
    # SET WRITE DIRECTORY
    outDir = "ML_results/" + thisModelName
    if not Path(outDir).exists():
      os.makedirs(outDir)
    plt.savefig(outDir + "/" + x + "_vs_" + y + '.png', bbox_inches='tight')
    print(outDir + "/" + x + "_vs_" + y + ".png Created")
  else:
    print("Missconfigured show")

def evaluateCLModel(thisModelName, classifier, features, target):
  # SET WRITE DIRECTORY
  outDir = "ML_results/" + thisModelName
  if not Path(outDir).exists():
    os.makedirs(outDir)

  scores = model_selection.cross_val_score(
    classifier, features, target, scoring='accuracy', cv=50)

  # Write Evaluation Results
  fRes = open(outDir +"/Evaluation.txt", 'w+')
  fRes.write('{:>5} \n'.format('Model:' + thisModelName))
  fRes.write("Mean of Scores : " + str(scores.mean()) + "\n")
  for i, s in enumerate(scores):
    fRes.write('{:>10} {:>10}\n'.format(str(i) + ': ', str(s)))
  fRes.close()
  print(outDir +"/Evaluation.txt Created")

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