def iniStudnt(Rut, Year):
  return {
    "Rut" : Rut, "Year": Year,

    "S1_INS": 0, "S1_DRP": 0, "S1_BAD": 0, "S1_CVL": 0, 
    "S1_DID": 0, "S1_APVD": 0, "S1_RPVD": 0, 
    "S1_GRD_1TO19": 0, "S1_GRD_2TO29": 0, "S1_GRD_3TO39": 0, 
    "S1_GRD_4TO49": 0, "S1_GRD_5TO59": 0, "S1_GRD_6TO7": 0, 
    "S1_BEST_GRD": 0, "S1_WORST_GRD": 0, "S1_AVG_PERF": 0, 

    "S2_INS": 0, "S2_DRP": 0, "S2_BAD": 0, "S2_CVL": 0, 
    "S2_DID": 0, "S2_APVD": 0, "S2_RPVD": 0, 
    "S2_GRD_1TO19": 0, "S2_GRD_2TO29": 0, "S2_GRD_3TO39": 0, 
    "S2_GRD_4TO49": 0, "S2_GRD_5TO59": 0, "S2_GRD_6TO7": 0, 
    "S2_BEST_GRD": 0, "S2_WORST_GRD": 0, "S2_AVG_PERF": 0, 

    "S2_VS_S1": 0}

def consolidate(Studnt):
  # Semester 1
  Studnt["S1_DID"] = Studnt["S1_INS"] - Studnt["S1_DRP"] - Studnt["S1_BAD"] - Studnt["S1_CVL"]
  Studnt["S1_AVG_PERF"] = 0 if Studnt["S1_DID"] == 0 else round(Studnt["S1_AVG_PERF"] / Studnt["S1_DID"],2)
  # Semester 2
  Studnt["S2_DID"] = Studnt["S2_INS"] - Studnt["S2_DRP"] - Studnt["S2_BAD"] - Studnt["S2_CVL"]
  Studnt["S2_AVG_PERF"] = 0 if Studnt["S2_DID"] == 0 else round(Studnt["S2_AVG_PERF"] / Studnt["S2_DID"],2)
  # Contrast
  Studnt["S2_VS_S1"] = Studnt["S2_AVG_PERF"] - Studnt["S1_AVG_PERF"]
  return Studnt

def reOrderCols(dfEnrolls):
  dfEnrolls_cols = [
  "Rut", "NEMScr", "Ranking", "LangScr", "MathScr", "ScienScr", 
  "SchoolRegion", "EdTypeCode", "SchoolType", 
  "MotherEd", "Campus", "PostulationType", 
  "S1_INS", "S1_DRP", "S1_BAD", "S1_CVL", "S1_DID", "S1_APVD", "S1_RPVD", 
  "S1_GRD_1TO19", "S1_GRD_2TO29", "S1_GRD_3TO39", 
  "S1_GRD_4TO49", "S1_GRD_5TO59", "S1_GRD_6TO7", 
  "S1_BEST_GRD", "S1_WORST_GRD", "S1_AVG_PERF", 
  "S2_INS", "S2_DRP", "S2_BAD", "S2_CVL", "S2_DID", "S2_APVD", "S2_RPVD", 
  "S2_GRD_1TO19", "S2_GRD_2TO29", "S2_GRD_3TO39", 
  "S2_GRD_4TO49", "S2_GRD_5TO59", "S2_GRD_6TO7", 
  "S2_BEST_GRD", "S2_WORST_GRD", "S2_AVG_PERF", 
  "S2_VS_S1", "Desertor"]
  return dfEnrolls[dfEnrolls_cols]

def aggregateEnrollwGrades(dfEnrolls, dfGrades, debug):
  # VARS
  Population = []
  Studnt = iniStudnt(0, 0)

  for idx, row in dfGrades.iterrows():
    if dfEnrolls['Rut'].isin([row['Rut']]).count() > 0:
      if Studnt["Rut"] == 0:
        Studnt["Rut"] = dfGrades.loc[idx,'Rut']
        Studnt["Year"] = dfGrades.loc[idx,'Year']
      elif Studnt["Rut"] != dfGrades.loc[idx,'Rut']:
        # print(Studnt)
        Studnt = consolidate(Studnt)
        Population.append(Studnt)
        Studnt = iniStudnt(dfGrades.loc[idx,'Rut'], dfGrades.loc[idx,'Year'])
      
      if dfGrades.loc[idx,'Year'] == Studnt["Year"]:
        # Semester 1
        if dfGrades.loc[idx,'Period'] == 1:
          Studnt["S1_INS"] += 1
          if 10 <= dfGrades.loc[idx,'Performance'] <= 70:
            if dfGrades.loc[idx,'Performance'] < 40:
              Studnt["S1_RPVD"] += 1
              if dfGrades.loc[idx,'Performance'] >= 30:
                Studnt["S1_GRD_3TO39"] += 1
              elif dfGrades.loc[idx,'Performance'] >= 20:
                Studnt["S1_GRD_2TO29"] += 1
              elif dfGrades.loc[idx,'Performance'] >= 10:
                Studnt["S1_GRD_1TO19"] += 1
            else:
              Studnt["S1_APVD"] += 1
              if dfGrades.loc[idx,'Performance'] < 50:
                Studnt["S1_GRD_4TO49"] += 1
              elif dfGrades.loc[idx,'Performance'] < 60:
                Studnt["S1_GRD_5TO59"] += 1
              elif dfGrades.loc[idx,'Performance'] <= 70:
                Studnt["S1_GRD_6TO7"] += 1
            
            if Studnt["S1_BEST_GRD"] == 0:
              Studnt["S1_BEST_GRD"] = dfGrades.loc[idx,'Performance']
            elif Studnt["S1_BEST_GRD"] < dfGrades.loc[idx,'Performance']:
              Studnt["S1_BEST_GRD"] = dfGrades.loc[idx,'Performance']

            if Studnt["S1_WORST_GRD"] == 0:
              Studnt["S1_WORST_GRD"] = dfGrades.loc[idx,'Performance']
            elif Studnt["S1_WORST_GRD"] > dfGrades.loc[idx,'Performance']:
              Studnt["S1_WORST_GRD"] = dfGrades.loc[idx,'Performance']

            Studnt["S1_AVG_PERF"] += dfGrades.loc[idx,'Performance']

          elif dfGrades.loc[idx,'Performance'] == 991: #'DRP'
            Studnt["S1_DRP"] += 1
          elif dfGrades.loc[idx,'Performance'] == 992: #'BAD'
            Studnt["S1_BAD"] += 1
          elif dfGrades.loc[idx,'Performance'] == 993: #'CVL'
            Studnt["S1_CVL"] += 1
        # Semester 2   
        elif dfGrades.loc[idx,'Period'] == 2:
          Studnt["S2_INS"] += 1
          if 10 <= dfGrades.loc[idx,'Performance'] <= 70:
            if dfGrades.loc[idx,'Performance'] < 40:
              Studnt["S2_RPVD"] += 1
              if dfGrades.loc[idx,'Performance'] >= 30:
                Studnt["S2_GRD_3TO39"] += 1
              elif dfGrades.loc[idx,'Performance'] >= 20:
                Studnt["S2_GRD_2TO29"] += 1
              elif dfGrades.loc[idx,'Performance'] >= 10:
                Studnt["S2_GRD_1TO19"] += 1
            else:
              Studnt["S2_APVD"] += 1
              if dfGrades.loc[idx,'Performance'] < 50:
                Studnt["S2_GRD_4TO49"] += 1
              elif dfGrades.loc[idx,'Performance'] < 60:
                Studnt["S2_GRD_5TO59"] += 1
              elif dfGrades.loc[idx,'Performance'] <= 70:
                Studnt["S2_GRD_6TO7"] += 1
            
            if Studnt["S2_BEST_GRD"] == 0:
              Studnt["S2_BEST_GRD"] = dfGrades.loc[idx,'Performance']
            elif Studnt["S2_BEST_GRD"] < dfGrades.loc[idx,'Performance']:
              Studnt["S2_BEST_GRD"] = dfGrades.loc[idx,'Performance']

            if Studnt["S2_WORST_GRD"] == 0:
              Studnt["S2_WORST_GRD"] = dfGrades.loc[idx,'Performance']
            elif Studnt["S2_WORST_GRD"] > dfGrades.loc[idx,'Performance']:
              Studnt["S2_WORST_GRD"] = dfGrades.loc[idx,'Performance']

            Studnt["S2_AVG_PERF"] += dfGrades.loc[idx,'Performance']

          elif dfGrades.loc[idx,'Performance'] == 991: #'DRP'
            Studnt["S2_DRP"] += 1
          elif dfGrades.loc[idx,'Performance'] == 992: #'BAD'
            Studnt["S2_BAD"] += 1
          elif dfGrades.loc[idx,'Performance'] == 993: #'CVL'
            Studnt["S2_CVL"] += 1
    else:
      print("NOT Found: ", row['Rut'])

  # Append Last Studnt
  Studnt = consolidate(Studnt)
  Population.append(Studnt)

  # Add Aggregations to enrolls
  for idx, row in dfEnrolls.iterrows():
    Studnt = list(filter(lambda Studnt: Studnt['Rut'] == row['Rut'], Population))
    if Studnt:
      dfEnrolls.loc[idx,'S1_INS'] = Studnt[0]['S1_INS']
      dfEnrolls.loc[idx,'S1_DRP'] = Studnt[0]['S1_DRP']
      dfEnrolls.loc[idx,'S1_BAD'] = Studnt[0]['S1_BAD']
      dfEnrolls.loc[idx,'S1_CVL'] = Studnt[0]['S1_CVL']
      dfEnrolls.loc[idx,'S1_DID'] = Studnt[0]['S1_DID']
      dfEnrolls.loc[idx,'S1_APVD'] = Studnt[0]['S1_APVD']
      dfEnrolls.loc[idx,'S1_RPVD'] = Studnt[0]['S1_RPVD']
      dfEnrolls.loc[idx,'S1_GRD_1TO19'] = Studnt[0]['S1_GRD_1TO19']
      dfEnrolls.loc[idx,'S1_GRD_2TO29'] = Studnt[0]['S1_GRD_2TO29']
      dfEnrolls.loc[idx,'S1_GRD_3TO39'] = Studnt[0]['S1_GRD_3TO39']
      dfEnrolls.loc[idx,'S1_GRD_4TO49'] = Studnt[0]['S1_GRD_4TO49']
      dfEnrolls.loc[idx,'S1_GRD_5TO59'] = Studnt[0]['S1_GRD_5TO59']
      dfEnrolls.loc[idx,'S1_GRD_6TO7'] = Studnt[0]['S1_GRD_6TO7']
      dfEnrolls.loc[idx,'S1_BEST_GRD'] = Studnt[0]['S1_BEST_GRD']
      dfEnrolls.loc[idx,'S1_WORST_GRD'] = Studnt[0]['S1_WORST_GRD']
      dfEnrolls.loc[idx,'S1_AVG_PERF'] = Studnt[0]['S1_AVG_PERF']

      dfEnrolls.loc[idx,'S2_INS'] = Studnt[0]['S2_INS']
      dfEnrolls.loc[idx,'S2_DRP'] = Studnt[0]['S2_DRP']
      dfEnrolls.loc[idx,'S2_BAD'] = Studnt[0]['S2_BAD']
      dfEnrolls.loc[idx,'S2_CVL'] = Studnt[0]['S2_CVL']
      dfEnrolls.loc[idx,'S2_DID'] = Studnt[0]['S2_DID']
      dfEnrolls.loc[idx,'S2_APVD'] = Studnt[0]['S2_APVD']
      dfEnrolls.loc[idx,'S2_RPVD'] = Studnt[0]['S2_RPVD']
      dfEnrolls.loc[idx,'S2_GRD_1TO19'] = Studnt[0]['S2_GRD_1TO19']
      dfEnrolls.loc[idx,'S2_GRD_2TO29'] = Studnt[0]['S2_GRD_2TO29']
      dfEnrolls.loc[idx,'S2_GRD_3TO39'] = Studnt[0]['S2_GRD_3TO39']
      dfEnrolls.loc[idx,'S2_GRD_4TO49'] = Studnt[0]['S2_GRD_4TO49']
      dfEnrolls.loc[idx,'S2_GRD_5TO59'] = Studnt[0]['S2_GRD_5TO59']
      dfEnrolls.loc[idx,'S2_GRD_6TO7'] = Studnt[0]['S2_GRD_6TO7']
      dfEnrolls.loc[idx,'S2_BEST_GRD'] = Studnt[0]['S2_BEST_GRD']
      dfEnrolls.loc[idx,'S2_WORST_GRD'] = Studnt[0]['S2_WORST_GRD']
      dfEnrolls.loc[idx,'S2_AVG_PERF'] = Studnt[0]['S2_AVG_PERF']

      dfEnrolls.loc[idx,'S2_VS_S1'] = Studnt[0]['S2_VS_S1']
    else:
      print("NOT Found: ", row['Rut'])

  return reOrderCols(dfEnrolls)

