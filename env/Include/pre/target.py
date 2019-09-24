def getTarget(dfEnrolls, dfGrades, sample):
  # Target : Desertor
  # 'y' variable aka target/outcome
  # Student which does not have subject after 1 year are considered deserter
  grades_maxYear = dfGrades.groupby('Rut').max()['Year']
  grades_minYear = dfGrades.groupby('Rut').min()['Year']

  for idx, row in dfEnrolls.iterrows():
    if row['Rut'] in grades_maxYear.keys():
      if sample:
        print("Rut: " + str(row['Rut']) + " Max: " + str(grades_maxYear[row['Rut']]) + " Min: " + str(grades_minYear[row['Rut']]) )
      dfEnrolls.loc[idx,'Desertor'] = 1 if grades_maxYear[row['Rut']] - grades_minYear[row['Rut']] == 0 else 0
    else:
      print("Rut NOT found: " + str(row['Rut']))
      dfEnrolls.loc[idx,'Desertor'] = -1
  return dfEnrolls, dfGrades

