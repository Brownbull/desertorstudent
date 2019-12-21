# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  PP
"""
import os
from pathlib import Path
from env.Include.dash.imports import *

def PP_exec(dashName, dataset, hue, vars, outF):
  if not Path(outF).exists():
    os.makedirs(outF)

  pp = sns.pairplot(  
    dataset, 
    hue = hue, 
    vars = vars, 
    size= 1.2 ) 
  pp.savefig(outF + dashName +".png")
  print("File Saved: " +  outF + dashName +".png")