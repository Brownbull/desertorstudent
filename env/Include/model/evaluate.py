import os
from pathlib import Path

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
  fRes.write('{:>10} {:>10}\n'.format('y:', modelResults['y']))
  fRes.write('{:>15}  {:>20}\n'.format('Accuracy:', Accuracy))
  fRes.write('{:>15}  {:>20}\n'.format('Precision:', Precision))
  fRes.write('{:>15}  {:>20}\n'.format('Recall:', Recall))
  fRes.write('{:>15}  {:>20}\n'.format('F1_Score:', F1_Score))
  fRes.close()
  print(outDir +"/Evaluation.txt Created")

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
  print(outDir +"/EvaluationCL.txt Created")