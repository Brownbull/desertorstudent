# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  ML - Machine Learning Module
"""
# IMPORT LIBRARIES
from env.Include.lib.functions import *
from env.Include.ml.optimize import *
from env.Include.ml.processing import *
from env.Include.ml.visual import *
from env.Include.ml.evaluate import *
from env.Include.ml.imports import *
# import env.Include.ml.features as x

# CHECK ARGUMENTS
parser = argparse.ArgumentParser(description='Main process of ML implementation to estimate rate of student desertion.')
parser.add_argument('-mlConfig','-mlc', '-c', help='ML Config File Path', default="config/MLconfig.yaml")
args = parser.parse_args()

# READ CONFIG FILE
mlCfg = readConfg(args.mlConfig)
if mlCfg['debug']: print(mlCfg)

# LIB INFO
if mlCfg['debug']: 
  getVersions()
  print("Debug Options: \n", args)

# START TIMING
timeStart = time.time()
dtStart = datetime.fromtimestamp(timeStart)
print("\nMain Script Start: " + str(dtStart) + "\n" + "-"*25 )

# GET INPUT DATA
if mlCfg['sample']:
  dataset = pd.read_csv(mlCfg['dataset'], nrows = mlCfg['sample'])
else:
  dataset = pd.read_csv(mlCfg['dataset'])

# SET FEATURES
X = dataset[mlCfg['ID'] + mlCfg['cat_enroll'] + mlCfg['num_PSU'] + mlCfg['num_S1'] + mlCfg['num_S2']]
# SET TARGET 
Y = dataset[mlCfg['Target']]

# ENCODE DATA
X_see = cat2Dummy(X, mlCfg['cat_enroll'])
X_bin = cat2Number(X, mlCfg['cat_enroll'])
X_enc = cat2Dummy(X_bin, mlCfg['cat_enroll'])

# end stage
finishedStage = "ML_01_ENCODE"
stageEnd(finishedStage, X_see, mlCfg['info'], mlCfg['debug'])
stageEnd(finishedStage, X_bin, mlCfg['info'], mlCfg['debug'])
stageEnd(finishedStage, X_enc, mlCfg['info'], mlCfg['debug'])

# STORE DATA
idx = False
saveFullDF(X_see, finishedStage, idx)
saveFullDF(X_bin, finishedStage, idx)
saveFullDF(X_enc, finishedStage, idx)

# RESET 
# SET FEATURES
X = dataset[mlCfg['ID'] + mlCfg['cat_enroll'] + mlCfg['num_PSU'] + mlCfg['num_S1'] + mlCfg['num_S2']]
# SET TARGET 
Y = dataset[mlCfg['Target']]

# CALL ML MODELS
from env.Include.ml.SLR import *
from env.Include.ml.MLR import *
from env.Include.ml.DT import *
from env.Include.ml.Kmeans import *
from env.Include.ml.PCA import *

reqModls = mlCfg['models']
traindMdls = {}
OutDir = "results/ML/"

# TRAIN MODELS
for config in reqModls:
  modelType = config['type']
  print("Processing model type:", modelType)

  # SLR
  if modelType.upper() == 'SLR':
    # VALIDATE CONFG
    if checkIfexists('x', config) and checkIfexists('y', config) and checkIfexists('show', config):
      # Model Name & Folder Path
      modelName = "SLR_" + config['x'] + "_vs_" + config['y']
      folderPath = OutDir + modelName + "/"
      # INPUT
      datasets = SLR_input(folderPath, modelName, X, config)
      # TRAIN & PREDICT
      traindMdls[modelName] = SLR_train(folderPath, modelName, datasets, config)
      # EVALUATE
      evaluateRegModel(
        traindMdls[modelName]['test_y'], traindMdls[modelName]['pred_y'], 
        modelName, traindMdls[modelName])
    else:
      # Conf Error
      print("Config in error for model: " + modelName)
  
  # MLR
  elif modelType.upper() == 'MLR':
    # VALIDATE CONFG
    if checkIfexists('x', config) and checkIfexists('y', config) and checkIfexists('show', config) and checkIfexists('xCategorical', config) and checkIfexists('xColNames', config):
      # Model Name & Folder Path
      modelName = "MLR_" + config['xColNames'] + "_vs_" + config['y']
      # TRAIN & PREDICT
      traindMdls[modelName] = MLR_train(modelName, X, Y, config)
      print("MLR Xcols")
      print(traindMdls[modelName]['x'])
      # EVALUATE
      evaluateRegModel(
        traindMdls[modelName]['test_y'], traindMdls[modelName]['pred_y'], 
        modelName, traindMdls[modelName])
    else:
      # Conf Error
      print("Config in error for model: " + modelName)

  # Decision Tree
  elif modelType.upper() == 'DT':
    if checkIfexists('x', config) and checkIfexists('y', config) and checkIfexists('show', config):
      # Model Name & Folder Path
      modelName = "DT_" + config['xColNames'] + "_vs_" + config['y']
      # TRAIN & PREDICT
      traindMdls[modelName]  = DT_train(modelName, X, Y, config)
      # EVALUATE
      evaluateRegModel(
        traindMdls[modelName]['test_y'], traindMdls[modelName]['pred_y'], 
        modelName, traindMdls[modelName])
    else:
      # Conf Error
      print("Config in error for model: " + modelName)

  # KMEANS
  elif modelType.upper() == 'KMEANS':
    # VALIDATE CONFG
    if checkIfexists('x', config) and checkIfexists('y', config) and checkIfexists('show', config):
      # Model Name & Folder Path
      modelName = "Kmeans_" + config['x'] + "_vs_" + config['y']
      # ELBOW
      n_clusters = KMeans_elbow(OutDir + modelName + "/", modelName, dataset, config)
      # TRAIN & PREDICT
      traindMdls[modelName] = KMeans_train(OutDir + modelName + "/", modelName, dataset, config, n_clusters)
    else:
      # Conf Error
      print("Config in error for model: " + modelName)
  
 
  # Model not listed    
  else:
    print("Not Recognized Model: " + modelType)

# SHOW TRAINED MODELS
print("*"*25,"\nTrained Models:")
for m in traindMdls.keys():
  print(m)

# END TIMING
timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nMain Script End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
# TEMP FILE END
# fTemp.close()
