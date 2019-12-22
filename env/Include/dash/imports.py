# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  Imports for Dashboards modules
"""
# misc libraries
import os
import sys #access to system parameters https://docs.python.org/3/library/sys.html
import argparse
import random
import time
from datetime import datetime

# load packages
import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features
import matplotlib #collection of functions for scientific and publication-ready visualization
import numpy as np #foundational package for scientific computing
import scipy as sp #collection of functions for scientific computing and advance mathematics
from IPython import display #pretty printing of dataframes in Jupyter notebook
import IPython
import sklearn #collection of machine learning algorithms

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.plotting import scatter_matrix

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

def getVersions():
  print("Python version: {}". format(sys.version))
  print("pandas version: {}". format(pd.__version__))
  print("matplotlib version: {}". format(matplotlib.__version__))
  print("NumPy version: {}". format(np.__version__))
  print("SciPy version: {}". format(sp.__version__)) 
  print("IPython version: {}". format(IPython.__version__)) 
  print("scikit-learn version: {}". format(sklearn.__version__))

def graphCongInit():
  #Configure Visualization Defaults
  #%matplotlib inline = show plots in Jupyter Notebook browser
  #%matplotlib inline
  mpl.style.use('ggplot')
  sns.set_style('white')
  pylab.rcParams['figure.figsize'] = 12,8