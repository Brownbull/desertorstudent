# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
"""
# IMPORT LIBRARIES
from env.Include.lib.functions import *
from env.Include.model.tools import *
from env.Include.model.processing import *
from env.Include.model.visual import *
from env.Include.model.evaluate import *
from env.Include.model.imports import *
# import env.Include.model.features as x

# CHECK ARGUMENTS
parser = argparse.ArgumentParser(description='Main process of ML implementation to estimate rate of student desertion.')
parser.add_argument('-mlConfig','-mlc', '-c', help='ML Config File Path', default="MLconfig.yaml")
args = parser.parse_args()

# READ CONFIG FILE
mlCfg = readMLConfg(args.mlConfig)
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
finishedStage = "10_ENCODE"
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
from env.Include.model.SLR import *
from env.Include.model.MLR import *
from env.Include.model.DT import *
from env.Include.model.Kmeans import *

reqModls = mlCfg['models']
traindMdls = {}

# TRAIN MODELS
for config in reqModls:
  modelType = config['type']
  print("Processing model type:", modelType)
  # SLR
  if modelType in ['slr', 'SLR']:
    # VALIDATE CONFG
    if checkIfexists('x', config) and checkIfexists('y', config) and checkIfexists('show', config):
      # Model Name
      modelName = "SLR_" + config['x'] + "_vs_" + config['y']
      # TRAIN
      traindMdls[modelName] = SLR_train(modelName, X_enc, config)
      # EVALUATE
      evaluateRegModel(
        traindMdls[modelName]['test_y'], traindMdls[modelName]['pred_y'], 
        modelName, traindMdls[modelName])
    else:
      # Conf Error
      print("Config in error for model: " + modelName)
  
  # MLR
  elif modelType in ['mlr', 'MLR']:
    # VALIDATE CONFG
    if checkIfexists('x', config) and checkIfexists('y', config) and checkIfexists('show', config) and checkIfexists('xCategorical', config) and checkIfexists('xColNames', config):
      # Model Name
      modelName = "MLR_" + config['xColNames'] + "_vs_" + config['y']
      # TRAIN
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
  elif modelType in ['dt', 'DT']:
    if checkIfexists('x', config) and checkIfexists('y', config) and checkIfexists('show', config):
      # Model Name
      modelName = "DT_" + config['xColNames'] + "_vs_" + config['y']
      # TRAIN
      traindMdls[modelName]  = DT_train(modelName, X, Y, config)
      # EVALUATE
      evaluateRegModel(
        traindMdls[modelName]['test_y'], traindMdls[modelName]['pred_y'], 
        modelName, traindMdls[modelName])
    else:
      # Conf Error
      print("Config in error for model: " + modelName)

  # Kmeans
  elif modelType in ['kmeans', 'Kmeans', 'KMEANS']:
    # VALIDATE CONFG
    if checkIfexists('x', config) and checkIfexists('y', config) and checkIfexists('show', config):
      # Model Name
      modelName = "Kmeans_" + config['x'] + "_vs_" + config['y']
      # TRAIN
      # traindMdls[modelName] = KMeans_train(modelName, dataset, config)
      # # EVALUATE
      # evaluateRegModel(
      #   traindMdls[modelName]['test_y'], traindMdls[modelName]['pred_y'], 
        # modelName, traindMdls[modelName])
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
