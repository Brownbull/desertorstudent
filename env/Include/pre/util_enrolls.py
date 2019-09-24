import numpy as np
import pandas as pd

def enrolls_Format(dfEnrolls):
  # RENAME COLS
  dfEnrolls.columns = ['EntryYear', 'TypeId', 'Rut', 'PlanId', 'DemreCode', 'Career', 'Campus', 'PostulationType', 'EntryType', 'NEM', 'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 'HistScr', 'PrefUM', 'PrefDemre', 'SchoolCity', 'SchoolRegion', 'EdTypeCode', 'EdType', 'SchoolType', 'MotherEd']
  # REMOVE LAST RUT CHAR
  dfEnrolls['Rut'] = dfEnrolls['Rut'].astype(str).str[:-1]
  dfEnrolls['Rut'] = pd.to_numeric(dfEnrolls['Rut'])
  # SORT FILES
  dfEnrolls = dfEnrolls.sort_values(by=['Rut', 'EntryYear'])
  # DROP DUPLICATES
  dfEnrolls = dfEnrolls.drop_duplicates(subset=['EntryYear', 'TypeId', 'Rut', 'DemreCode', 'Career', 'Campus', 'PostulationType', 'NEM', 'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 'HistScr', 'PrefUM', 'PrefDemre', 'SchoolRegion', 'EdTypeCode', 'EdType', 'SchoolType', 'MotherEd'], keep='last')
  # DROP COLUMNS
  enrolls_drop_cols = ['TypeId','PlanId','DemreCode','Career','EntryType','NEM','HistScr','PrefUM','PrefDemre','SchoolCity','EdType']
  dfEnrolls = dfEnrolls.drop(enrolls_drop_cols, axis=1)
  
  return dfEnrolls

def enrolls_Fill(dfEnrolls):
  # Tried to use assign & replace pandas functions but columns types do not help
  """
  Enrolls Fills
  Ranking : median
  NEMScr : median
  LangScr: median
  MathScr: median
  ScienScr: median
  SchoolRegion: mode
  EdTypeCode: mode
  SchoolType: mode
  MotherEd: mode
  """
  mean_NEMScr = round(dfEnrolls['NEMScr'].mean(skipna=True),0)
  mean_Ranking = round(dfEnrolls['Ranking'].mean(skipna=True),0)
  mean_LangScr = round(dfEnrolls['LangScr'].mean(skipna=True),0)
  mean_MathScr = round(dfEnrolls['MathScr'].mean(skipna=True),0)
  mean_ScienScr = round(dfEnrolls['ScienScr'].mean(skipna=True),0)

  dfEnrolls['NEMScr'].fillna(dfEnrolls['NEMScr'].median(), inplace = True)
  dfEnrolls['NEMScr']=dfEnrolls.NEMScr.mask(dfEnrolls.NEMScr == 0,mean_NEMScr)
  dfEnrolls['Ranking'].fillna(dfEnrolls['Ranking'].median(), inplace = True)
  dfEnrolls['Ranking']=dfEnrolls.Ranking.mask(dfEnrolls.Ranking == 0,mean_Ranking)
  dfEnrolls['LangScr'].fillna(dfEnrolls['LangScr'].median(), inplace = True)
  dfEnrolls['LangScr']=dfEnrolls.LangScr.mask(dfEnrolls.LangScr == 0,mean_LangScr)
  dfEnrolls['MathScr'].fillna(dfEnrolls['MathScr'].median(), inplace = True)
  dfEnrolls['MathScr']=dfEnrolls.MathScr.mask(dfEnrolls.MathScr == 0,mean_MathScr)
  dfEnrolls['ScienScr'].fillna(dfEnrolls['ScienScr'].median(), inplace = True)
  dfEnrolls['ScienScr']=dfEnrolls.ScienScr.mask(dfEnrolls.ScienScr == 0,mean_ScienScr)

  dfEnrolls['Ranking'].fillna(dfEnrolls['Ranking'].median(), inplace = True)
  dfEnrolls['NEMScr'].fillna(dfEnrolls['NEMScr'].median(), inplace = True)
  dfEnrolls['LangScr'].fillna(dfEnrolls['LangScr'].median(), inplace = True)
  dfEnrolls['MathScr'].fillna(dfEnrolls['MathScr'].median(), inplace = True)
  dfEnrolls['ScienScr'].fillna(dfEnrolls['ScienScr'].median(), inplace = True)

  dfEnrolls['SchoolRegion'].fillna(dfEnrolls['SchoolRegion'].mode()[0], inplace = True)
  dfEnrolls['EdTypeCode'].fillna(dfEnrolls['EdTypeCode'].mode()[0], inplace = True)
  dfEnrolls['SchoolType'].fillna(dfEnrolls['SchoolType'].mode()[0], inplace = True)
  dfEnrolls['MotherEd'].fillna(0, inplace = True) # 0: No information

  return dfEnrolls

def enrolls_FeatureEng(dfEnrolls):
  for idx, row in dfEnrolls.iterrows():
    # CampusStgo
    if dfEnrolls.loc[idx,'Campus'][0] in ['S', 's']:
      dfEnrolls.loc[idx,'Campus'] = 'S'
    else:
      dfEnrolls.loc[idx,'Campus'] = 'T'

    # PostulationRegular
    if dfEnrolls.loc[idx,'PostulationType'][0] in ['R', 'r']:
      dfEnrolls.loc[idx,'PostulationType'] = 'R'
    else:
      dfEnrolls.loc[idx,'PostulationType'] = 'E'

    # SchoolRegion
    if dfEnrolls['SchoolRegion'].dtype != np.float64: 
    # added for sample, 10 rows will case np.float64 tpe which does not have isnumeric() method
      if dfEnrolls.loc[idx,'SchoolRegion'] in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']:
        dfEnrolls.loc[idx,'SchoolRegion'] = int(dfEnrolls.loc[idx,'SchoolRegion'])
      else:
      # if not dfEnrolls.loc[idx,'SchoolRegion'].isnumeric():
        if dfEnrolls.loc[idx,'SchoolRegion'] == 'I':
          dfEnrolls.loc[idx,'SchoolRegion'] = 1
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'II':
          dfEnrolls.loc[idx,'SchoolRegion'] = 2
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'III':
          dfEnrolls.loc[idx,'SchoolRegion'] = 3
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'IV':
          dfEnrolls.loc[idx,'SchoolRegion'] = 4
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'V':
          dfEnrolls.loc[idx,'SchoolRegion'] = 5
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'VI':
          dfEnrolls.loc[idx,'SchoolRegion'] = 6
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'VII':
          dfEnrolls.loc[idx,'SchoolRegion'] = 7
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'VIII':
          dfEnrolls.loc[idx,'SchoolRegion'] = 8
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'IX':
          dfEnrolls.loc[idx,'SchoolRegion'] = 9
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'X':
          dfEnrolls.loc[idx,'SchoolRegion'] = 10
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'XI':
          dfEnrolls.loc[idx,'SchoolRegion'] = 11
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'XII':
          dfEnrolls.loc[idx,'SchoolRegion'] = 12
        elif dfEnrolls.loc[idx,'SchoolRegion'] in ['XIII', 'RM']:
          dfEnrolls.loc[idx,'SchoolRegion'] = 13
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'XIV':
          dfEnrolls.loc[idx,'SchoolRegion'] = 14
        elif dfEnrolls.loc[idx,'SchoolRegion'] == 'XV':
          dfEnrolls.loc[idx,'SchoolRegion'] = 15
        else:
          dfEnrolls.loc[idx,'SchoolRegion'] = dfEnrolls.loc[idx,'SchoolRegion'] + "_UNK"
    # EdTypeCode
    edType = dfEnrolls.loc[idx,'EdTypeCode'].strip()
    if edType == 'H1': # Humanista Científico Diurno
      dfEnrolls.loc[idx,'EdTypeCode'] = 'HCD'
    elif edType == 'H2': # Humanista Científico Nocturno
      dfEnrolls.loc[idx,'EdTypeCode'] = 'HCN'
    elif edType == 'T1': # Técnico Profesional Comercial
      dfEnrolls.loc[idx,'EdTypeCode'] = 'TPC'
    elif edType == 'T2': # Técnico Profesional Industrial
      dfEnrolls.loc[idx,'EdTypeCode'] = 'TPI'
    elif edType == 'T3': # Técnico Profesional Servicios y Técnica
      dfEnrolls.loc[idx,'EdTypeCode'] = 'TPS'
    elif edType == 'CEFT': # Centro de Formacion Tecnica
      dfEnrolls.loc[idx,'EdTypeCode'] = 'CFT'
    else:
      dfEnrolls.loc[idx,'EdTypeCode'] = 'UNK'

    # SchoolType
    if dfEnrolls.loc[idx,'SchoolType'] == 'Municipal':
      dfEnrolls.loc[idx,'SchoolType'] = 'MUN'
    elif dfEnrolls.loc[idx,'SchoolType'] == 'Particular Subvencionado':
      dfEnrolls.loc[idx,'SchoolType'] = 'PSS'
    elif dfEnrolls.loc[idx,'SchoolType'] in ['Particular no subvencionado', 'Particular NO Subvencionado']:
      dfEnrolls.loc[idx,'SchoolType'] = 'PNS'
    elif 'Delegada' in dfEnrolls.loc[idx,'SchoolType']:
      dfEnrolls.loc[idx,'SchoolType'] = 'CAD' # Corporacion Administracion Delegada
    else:
      dfEnrolls.loc[idx,'SchoolType'] = 'CMN' # Corporacion Municipal

  # Set columns as Objects
  cnvCatCols = ['Campus', 'PostulationType', 'EdTypeCode', 'SchoolType', 'MotherEd']
  for col in cnvCatCols:
    dfEnrolls[col] = dfEnrolls[col].astype(str)

  return dfEnrolls

def enrolls_DropCols(dfEnrolls):
  return dfEnrolls