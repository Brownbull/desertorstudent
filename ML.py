# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
"""
# IMPORT LIBRARIES
from env.Include.lib.functions import *
from env.Include.model.operations import *
from env.Include.model.imports_model import *
import env.Include.model.features as x

# define y variable aka target/outcome
ID = ['Rut']
y = 'Desertor'

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
X = dataset[ID + x.cat_enroll + x.num_PSU + x.num_S1 + x.num_S2]
# SET TARGET 
Y = dataset[y]

# ENCODE DATA
X_see = cat2Dummy(X, x.cat_enroll)
X_bin = cat2Number(X, x.cat_enroll)
X_enc = cat2Dummy(X_bin, x.cat_enroll)

# end stage
finishedStage = "10_ENCODE"
stageEnd(finishedStage, X_see, mlCfg['info'], mlCfg['debug'])
stageEnd(finishedStage, X_bin, mlCfg['info'], mlCfg['debug'])
stageEnd(finishedStage, X_enc, mlCfg['info'], mlCfg['debug'])
# save data
idx = False
saveFullDF(X_see, finishedStage, idx)
saveFullDF(X_bin, finishedStage, idx)
saveFullDF(X_enc, finishedStage, idx)

# CALL ML MODELS
from env.Include.model.SLR import *
from env.Include.model.MLR import *

requestedModels = mlCfg['models']
trainedModels = {}

# RESET 
# SET FEATURES
X = dataset[ID + x.cat_enroll + x.num_PSU + x.num_S1 + x.num_S2]
# SET TARGET 
Y = dataset[y]

# TRAIN MODELS
for rModel in requestedModels:
  modelType = rModel['type']
  # SLR
  if modelType in ['slr', 'SLR']:
    print("Processing model type:", modelType)
    if checkIfexists('x', rModel) and checkIfexists('y', rModel) and checkIfexists('show', rModel):
      # TRAIN
      model_SLR, thisModelName, test_y, pred_y = SLR_train(X_enc, rModel)
      # STORE RESULTS
      trainedModels[thisModelName] = {
        'config': rModel,
        'model': model_SLR,
        'x' : rModel['x'], 
        'y' : rModel['y'],
        'test_y' : test_y,
        'pred_y': pred_y
      }
      # EVALUATE
      evaluateRegModel(test_y, pred_y, thisModelName, trainedModels[thisModelName])
    else:
      # Conf Error
      print("Config in error for model: " + thisModelName)
  
  # MLR
  elif modelType in ['mlr', 'MLR']:
    print("Processing model type:", modelType)
    if checkIfexists('x', rModel) and checkIfexists('y', rModel) and checkIfexists('show', rModel) and checkIfexists('xCategorical', rModel) and checkIfexists('xColNames', rModel):
      # TRAIN
      model_MLR, thisModelName, test_y, pred_y, Xcols, cols2DropDesc = MLR_train(X, Y, rModel)
      print("MLR Xcols")
      print(Xcols)
      # STORE RESULTS
      trainedModels[thisModelName] = {
        'config': rModel,
        'model': model_MLR,
        'x' : Xcols, 
        'y' : rModel['y'],
        'test_y' : test_y,
        'pred_y': pred_y
      }
      # EVALUATE
      evaluateRegModel(test_y, pred_y, thisModelName, trainedModels[thisModelName])
    else:
      # Conf Error
      print("Config in error for model: " + thisModelName)

  # Model not listed    
  else:
    print("Not Recognized Model: " + modelType)

# SHOW TRAINED MODELS
print("*"*25,"\nTrained Models:")
for m in trainedModels.keys():
  print(m)

# END TIMING
timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nMain Script End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
# TEMP FILE END
# fTemp.close()
