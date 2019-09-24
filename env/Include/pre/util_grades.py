import pandas as pd 

def grades_Format(dfGrades):
  # RENAME COLS
  dfGrades.columns = ['Rut', 'StudyPlan', 'PlanName', 'Subject', 'SubjName', 'EnrollReason', 'ReasonDesc', 'SubjStatus', 'Year', 'Period', 'SubjValidate', 'Section', 'Grade', 'SubjHomologate' ]
  # REMOVE LAST RUT CHAR
  dfGrades['Rut'] = dfGrades['Rut'].astype(str).str[:-1]
  dfGrades['Rut'] = pd.to_numeric(dfGrades['Rut'])
  # SORT FILES
  dfGrades = dfGrades.sort_values(by=['Rut', 'Year', 'Period', 'Grade', 'SubjStatus'])
  # DROP DUPLICATES
  dfGrades = dfGrades.drop_duplicates(subset=['Rut', 'Subject', 'SubjName', 'EnrollReason', 'ReasonDesc', 'SubjStatus', 'Year', 'Period', 'SubjValidate', 'Section', 'Grade', 'SubjHomologate' ], keep='last')
  # DROP COLUMNS
  grades_drop_cols = ['StudyPlan','PlanName','Subject','SubjName','EnrollReason','ReasonDesc','Section']
  dfGrades.drop(grades_drop_cols, axis=1, inplace = True)
  # DROP ROWS WITH NULL NON DEDUCIBLE INFO
  dfGrades = dfGrades.dropna(subset=['Period'])

  return dfGrades

def grades_Fill(dfGrades):
  return dfGrades

def grades_FeatureEng(dfGrades):
  """
  Performance: from Grade, SubjHomologate, SubjValidate
  """
  for idx, row in dfGrades.iterrows():
    # Performance
    if dfGrades.loc[idx,'Grade'] not in ['A', 'NA'] and not (pd.isnull(row['Grade'])):
      if 0.0 <= float(dfGrades.loc[idx,'Grade']) <= 7.1 and dfGrades.loc[idx,'SubjStatus'] != 4 and dfGrades.loc[idx,'SubjHomologate'] != 'X':
        dfGrades.loc[idx,'Performance'] = int(float(dfGrades.loc[idx,'Grade']) * 10)
      elif dfGrades.loc[idx,'SubjStatus'] == 4:
        dfGrades.loc[idx,'Performance'] = 991 #'DRP'
      elif dfGrades.loc[idx,'SubjStatus'] == 3 and dfGrades.loc[idx,'SubjValidate'] != 'X' and dfGrades.loc[idx,'SubjHomologate'] != 'X':
        dfGrades.loc[idx,'Performance'] = 992 #'BAD'
      elif dfGrades.loc[idx,'SubjStatus'] == 2 and (dfGrades.loc[idx,'SubjValidate'] == 'X' or dfGrades.loc[idx,'SubjHomologate'] == 'X' ):
        dfGrades.loc[idx,'Performance'] = 993 #'CVL'
      elif dfGrades.loc[idx,'SubjStatus'] == 1:
        dfGrades.loc[idx,'Performance'] = 994 #'NULL'
      else:
        dfGrades.loc[idx,'Performance'] = 999 #'UNK'
    elif dfGrades.loc[idx,'SubjStatus'] == 4:
      dfGrades.loc[idx,'Performance'] = 991 #'DRP'
    elif dfGrades.loc[idx,'SubjStatus'] == 3:
      dfGrades.loc[idx,'Performance'] = 992 #'BAD'
    elif dfGrades.loc[idx,'SubjStatus'] == 2:
      dfGrades.loc[idx,'Performance'] = 993 #'CVL'
    elif dfGrades.loc[idx,'SubjStatus'] == 1:
      dfGrades.loc[idx,'Performance'] = 994 #'NULL'
    else:
      dfGrades.loc[idx,'Performance'] = 999 #'UNK'
  dfGrades['Performance'] = dfGrades['Performance'].astype(int)

  return dfGrades

def grades_DropCols(dfGrades):
  dfGrades.drop(['SubjStatus', 'SubjValidate','Grade','SubjHomologate'], axis=1, inplace = True)
  return dfGrades