import os
import sys
import yaml
import inspect
from pathlib import Path
# EXCEL
import xlwt
from xlwt.Workbook import *
import pandas as pd
from pandas import ExcelWriter
import xlsxwriter

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

def setOrCreatePath(outDir):
  # SET WRITE DIRECTORY
  if not Path(outDir).exists():
    os.makedirs(outDir)

def props(cls):   
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

def checkIfexists(key, tree):
  if (key in tree) and tree[key] is not None:
    return True
  return False

def checkIfManyExists(keys, tree):
  for key in keys:
    if checkIfexists(key, tree) == False:
      return False
  return True

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

# YAML CONSTRUCTORS
def join(loader, node):
  seq = loader.construct_sequence(node)
  return ''.join([str(i) for i in seq])

def readConfg(fConfig):
  # INIT FUNCTIONS
  yaml.add_constructor('!join', join)

  # GET README CONFIG
  if Path(fConfig).is_file():
    with open(fConfig, 'r') as configFile:
      return yaml.load(configFile)
  else:
    sys.exit('Error: File ' + fConfig + " was not found.")

# SAVE Dataframes on EXCEL format
def saveDFs2xlsx(folderPath, fileName, excelJson, idx, dataType):
  """
  This function will store data in a xlsx format
  Input: 
    folderPath: Path of FOLDER to store file
    fileName: NAME of file to store in folderPath, xlsx will be added
    excelJson -> Example:
      excelJson = [
        {
          "sheetName": 'train',
          "sheetData": [ train_X, train_y ]
        },
        {
          "sheetName": 'test',
          "sheetData": [ test_X, test_y ]
        }
      ]
    idx: flag for Index column in excel
    dataType: Data Type to write in excel, can be:
      df -> dataframe
      rows -> data to write row by row
      columns -> data to write column by column
  """
  # SET WRITE VARS
  setOrCreatePath(folderPath)
  fileCreated = False
  xlsxName = folderPath + fileName + ".xlsx"

  # WRITE EXCEL
  for sheet in excelJson:
    if dataType.upper() == "DF":
      writer = pd.ExcelWriter(xlsxName, engine='xlsxwriter')
      for i, data in enumerate(sheet['sheetData']):
        data.to_excel(writer, sheet_name=sheet['sheetName'], startcol=i, index=idx)
      writer.save()
      fileCreated = True
    elif dataType.upper() == "ROWS":
      workbook =xlsxwriter.Workbook(xlsxName)
      worksheet = workbook.add_worksheet(sheet['sheetName'])
      row = 0
      col = 0
      for i, data in enumerate(sheet['sheetData']):
        worksheet.write_row(row + i, col, tuple(data))
      workbook.close()
      fileCreated = True

  # TELL RESULTS    
  if fileCreated:
    print("File Created: " + xlsxName)
  else:
    print("File NOT Created: " + xlsxName)
