# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
"""
# IMPORT LIBRARIES
from env.Include.lib.functions import *
from env.Include.pre.imports_pre import *
from env.Include.pre.util_grades import *
from env.Include.pre.util_enrolls import *
from env.Include.pre.target import *
from env.Include.pre.aggregate import * 

# CHECK ARGUMENTS
parser = argparse.ArgumentParser(description='ETL process of ML implementation to estimate rate of student desertion.')
parser.add_argument('-enrolls','-en', '-e', required=False, help='Enrolls.csv file path', default="data/Enrolls.csv")
parser.add_argument('-grades','-gd', '-g', required=False, help='Grades.csv file path', default="data/Grades.csv")
parser.add_argument('-sample','-sm', '-sp', required=False, help='Numbers of records to take as sample', default=0)
parser.add_argument('-min','-mn', required=False, help='Min cases to be common', default=11)
parser.add_argument('-info','-i', action='store_const', const=True, default=True, help='Dataframes Information Flag.')
parser.add_argument('-force','-f', action='store_const', const=True, default=False, help='Force Flag, delete any output file in place.')
parser.add_argument('-stats','-s', action='store_const', const=True, default=True, help='Stats Flag')
parser.add_argument('-debug','-d', action='store_const', const=True, default=True, help='Debug Flag')
args = parser.parse_args()

if args.debug: 
  getVersions()
  print("Debug Options: \n", args)

# START TIMING
timeStart = time.time()
dtStart = datetime.fromtimestamp(timeStart)
print("\nPreprocessing Script Start: " + str(dtStart) + "\n" + "-"*25 )

# TEMP FILE INIT
# SET WRITE DIRECTORY
outDir = "tmp"
if not Path(outDir).exists():
  os.makedirs(outDir)
fTemp = open(outDir + "/ETLTemp.txt", 'w+')

# GET INPUT DATA
if args.sample:
  enrolls_raw = pd.read_csv(args.enrolls, nrows = args.sample)
  grades_raw = pd.read_csv(args.grades, nrows = args.sample)
else:
  enrolls_raw = pd.read_csv(args.enrolls) 
  grades_raw = pd.read_csv(args.grades)
dfs = [enrolls_raw, grades_raw] # set array of data

# end stage
finishedStage = "01_GET_RAW"
stageEndSet(finishedStage, dfs, args.info, args.debug)

# FORMAT DATA
enrolls_data = enrolls_Format(enrolls_raw)
grades_data = grades_Format(grades_raw)
dfs = [enrolls_data, grades_data] # reset array of data

# end stage
finishedStage = "02_FORMAT"
stageEndSet(finishedStage, dfs, args.info, args.debug)

# FILL NULL DATA
enrolls_data = enrolls_Fill(enrolls_data)
grades_data = grades_Fill(grades_data)

# end stage
finishedStage = "03_FILL"
stageEndSet(finishedStage, dfs, args.info, args.debug)

# FEATURE ENG
# X : CREATE AND SET FEATURES
enrolls_data = enrolls_FeatureEng(enrolls_data)
grades_data = grades_FeatureEng(grades_data)
# Y : TARGET - DESERTOR FLAG
enrolls_data, grades_data = getTarget(enrolls_data, grades_data, args.sample)
# DROP USED COLUMNS
enrolls_data = enrolls_DropCols(enrolls_data)
grades_data = grades_DropCols(grades_data)

# end stage
finishedStage = "04_FEATURE_ENG"
stageEndSet(finishedStage, dfs, args.info, args.debug)
# save data
idx = False
saveFullDF(enrolls_data, finishedStage, idx)
saveFullDF(grades_data, finishedStage, idx)

# UNIFY RARE CASES
unifyUncommon(enrolls_data, args.debug, min = args.min)
unifyUncommon(grades_data, args.debug, min = args.min)

# end stage
finishedStage = "05_UNIFY_RARE"
stageEndSet(finishedStage, dfs, args.info, args.debug)

# AGGREGATE
enrolls_data = aggregateEnrollwGrades(enrolls_data, grades_data, args.debug)
dfs = [enrolls_data, grades_data] # set array of data

# end stage
finishedStage = "06_AGGREGATE"
stageEndSet(finishedStage, dfs, args.info, args.debug)
# save data
idx = False
saveFullDF(enrolls_data, finishedStage, idx)
saveFullDF(grades_data, finishedStage, idx)


# END TIMING
timeEnd = time.time()
dtEnd = datetime.fromtimestamp(timeEnd)
print('-'*25 + "\nPreprocessing Script End:" + str(dtEnd) + '\n' + '-'*25 + "\nTotal Time: " + str(timeEnd - timeStart) + "\n")
# TEMP FILE END
fTemp.close()
