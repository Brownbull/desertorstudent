import os
import sys
import yaml
import inspect
from pathlib import Path

# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string/18425523
def retrieveName(var):
  """
  Gets the name of var. Does it from the out most frame inner-wards.
  :param var: variable to get name from.
  :return: string
  """
  for fi in reversed(inspect.stack()):
    names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
    if len(names) > 0:
      return names[0]

def props(cls):   
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

def checkIfexists(key, tree):
  if (key in tree) and tree[key] is not None:
    return True
  return False

def stageEnd(stageName, df, info, debug):
  dfName = retrieveName(df)
  if debug: print("-"*25 + "\n"+ stageName + " DONE\n" + "-"*25 )
  if info:
    dfStats(df, dfName, stageName)

def stageEndSet(stageName, dfs, info, debug):
  dfsNames = []
  for d in dfs: 
    dfsNames.append(retrieveName(d))
  if debug: print("-"*25 + "\n"+ stageName + " DONE\n" + "-"*25 )
  if info:
    for i, d in enumerate(dfs):
      dfStats(d, dfsNames[i], stageName)

def dfStats(df, dfName, stageName):
  # SET WRITE DIRECTORY
  outDir = "reports/" + stageName + "/" + dfName
  if not Path(outDir).exists():
    os.makedirs(outDir)

  # START
  print("-"*20)
  print("Stats INI: " + dfName + " after " + stageName) 

  # INFO
  fInfo = open(outDir + "/info.txt", 'w+')
  df.info(buf=fInfo)
  fInfo.close()
  print(outDir +"/info.txt Created")

  # DESCRIBE
  df.describe(include = 'all').to_csv(outDir +"/describe.csv")
  print(outDir +"/describe.csv Created")

  # NULLS
  fNull = open(outDir +"/nulls.txt", 'w+')
  fNull.write(dfName + ' columns with null values:\n')
  nulls = df.isnull().sum()
  for key,value in nulls.iteritems():
    # https://stackoverflow.com/questions/8234445/python-format-output-string-right-alignment
    fNull.write('{:>30}  {:>20}\n'.format(key, str(value)))
  fNull.close()
  print(outDir +"/nulls.txt Created")

  # 0s
  fNull = open(outDir +"/ceros.txt", 'w+')
  fNull.write(dfName + ' columns with null values:\n')
  ceros = (df == 0).sum(axis=0)
  for key,value in ceros.iteritems():
    # https://stackoverflow.com/questions/8234445/python-format-output-string-right-alignment
    fNull.write('{:>30}  {:>20}\n'.format(key, str(value)))
  fNull.close()
  print(outDir +"/ceros.txt Created")

  # END
  print("Stats END: " + dfName + " after " + stageName) 
  print("-"*20)

def saveFullDF(df, stageName, idx):
  dfName = retrieveName(df)
  # SET WRITE DIRECTORY
  outDir = "data/" + stageName
  if not Path(outDir).exists():
    os.makedirs(outDir)

  # WRTIE DF
  df.to_csv(outDir + "/" + dfName +  ".csv", index=idx)
  print("Writing... " + outDir + "/" + dfName +  ".csv Created")

# while small is arbitrary, we'll use the common minimum in statistics: 
# http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
def unifyUncommon(df, debug, **kwargs):
  dfName = retrieveName(df)
  # Default
  min = 10
  # kwargs setup
  for key, value in kwargs.items(): 
    if key == 'min':
      min = value
  if debug: print("Using min: " + str(min) + " on " + dfName)

  # Function
  # Get columns to check rare cases
  categoricalVariables = list(df.select_dtypes(include=['object']).columns.values)
  if debug: 
    print(dfName + " - columns checked for unify:")
    print(categoricalVariables)

  # Replace rare cases in each column
  for col in categoricalVariables:
    unifyValue = 'X' * df[col].str.len().max()
    varCounts = (df[col].value_counts() < min)
    df[col] = df[col].apply(lambda x: unifyValue if varCounts.loc[x] == True else x)

  return df

# YAML CONSTRUCTORS
def join(loader, node):
  seq = loader.construct_sequence(node)
  return ''.join([str(i) for i in seq])

def readMLConfg(mlConfig):
  # INIT FUNCTIONS
  yaml.add_constructor('!join', join)

  # GET README CONFIG
  if Path(mlConfig).is_file():
    with open(mlConfig, 'r') as configFile:
      return yaml.load(configFile)
  else:
    sys.exit('Error: File ' + mlConfig + " was not found.")