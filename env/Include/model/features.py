# COLUMN SETS FILE
everything = [
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 'SchoolRegion', 
  'EdTypeCode', 'SchoolType', 'MotherEd', 'Campus', 'PostulationType', 
  'S1_INS', 'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_DID', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD', 'S1_AVG_PERF', 
  'S2_INS', 'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_DID', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 'S2_AVG_PERF', 'S2_VS_S1']

num_all = [ 
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr', 
  'S1_INS', 'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_DID', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD', 'S1_AVG_PERF', 
  'S2_INS', 'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_DID', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 'S2_AVG_PERF', 'S2_VS_S1'
]

cat_enroll = [
  'SchoolRegion', 'EdTypeCode', 'SchoolType', 'MotherEd', 'Campus', 'PostulationType'
]

cEnroll_nPSU = [
  'SchoolRegion', 'EdTypeCode', 'SchoolType', 'MotherEd', 'Campus', 'PostulationType',
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr'
]


num_PSU = [ 
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr'
]

num_S1 = [ 
  'S1_DRP', 'S1_BAD', 'S1_CVL',  
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD'
]

num_S2 = [  
  'S2_DRP', 'S2_BAD', 'S2_CVL', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 
  'S2_VS_S1'
]

num_S1_plus = [ 
  'S1_INS', 'S1_DRP', 'S1_BAD', 'S1_CVL', 'S1_DID', 'S1_APVD', 'S1_RPVD', 
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD', 'S1_AVG_PERF'
]

num_S2_plus = [  
  'S2_INS', 'S2_DRP', 'S2_BAD', 'S2_CVL', 'S2_DID', 'S2_APVD', 'S2_RPVD', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 'S2_AVG_PERF', 
  'S2_VS_S1'
]

cEnroll_nPSU_nS1S2 = [
  'SchoolRegion', 'EdTypeCode', 'SchoolType', 'MotherEd', 'Campus', 'PostulationType',
  'NEMScr', 'Ranking', 'LangScr', 'MathScr', 'ScienScr',
  'S1_DRP', 'S1_BAD', 'S1_CVL',  
  'S1_GRD_1TO19', 'S1_GRD_2TO29', 'S1_GRD_3TO39', 
  'S1_GRD_4TO49', 'S1_GRD_5TO59', 'S1_GRD_6TO7', 
  'S1_BEST_GRD', 'S1_WORST_GRD',
  'S2_DRP', 'S2_BAD', 'S2_CVL', 
  'S2_GRD_1TO19', 'S2_GRD_2TO29', 'S2_GRD_3TO39', 
  'S2_GRD_4TO49', 'S2_GRD_5TO59', 'S2_GRD_6TO7', 
  'S2_BEST_GRD', 'S2_WORST_GRD', 
  'S2_VS_S1'
]
