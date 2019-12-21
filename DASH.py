# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
"""
# IMPORT LIBRARIES
from env.Include.lib.functions import *
from env.Include.dash.imports import *

# CHECK ARGUMENTS
parser = argparse.ArgumentParser(description='Main process of Dashboards to generate visualizations from obtained data.')
parser.add_argument('-dashConfig','-dc', '-c', help='Dashboards Config File Path', default="config/DASHconfig.yaml")
args = parser.parse_args()

# READ CONFIG FILE
dashConfig = readConfg(args.dashConfig)
if dashConfig['debug']: print(dashConfig)

# LIB INFO
if dashConfig['debug']: 
  getVersions()
  print("Debug Options: \n", args)

# START TIMING
timeStart = time.time()
dtStart = datetime.fromtimestamp(timeStart)
print("\nMain Script Start: " + str(dtStart) + "\n" + "-"*25 )

# GET INPUT DATA
if dashConfig['sample']:
  dataset = pd.read_csv(dashConfig['dataset'], nrows = dashConfig['sample'])
else:
  dataset = pd.read_csv(dashConfig['dataset'])

# SET FEATURES
X = dataset[dashConfig['ID'] + dashConfig['cat_enroll'] + dashConfig['num_PSU'] + dashConfig['num_S1'] + dashConfig['num_S2']]
# SET TARGET 
Y = dataset[dashConfig['Target']]

# CREATE DASHBOARDS
from env.Include.dash.pp import *

reqDashboards = dashConfig['dashboards']
outF = "results/DASH/"

# EXECUTE DASH CREATE
for config in reqDashboards:
  dashType = config['type']
  print("Processing dashboard type:", dashType)
  # PP
  if dashType.upper() in ['PP']:
    # VALIDATE CONFG
    if checkIfManyExists(['xColNames', 'x', 'y'], config):
      # Exec
      dashName = "PP_" + config['xColNames'] + "_on_" + config['y']
      PP_exec(dashName, dataset, dashConfig['Target'], config['x'], outF)
    else:
      # Conf Error
      print("Config in error for dashboard: " + dashType)

# END TIMING
timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nMain Script End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
# TEMP FILE END
# fTemp.close()




