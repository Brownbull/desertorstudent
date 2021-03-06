# -*- coding: utf-8 -*-
"""
  @author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  Visualization functions
"""
from pathlib import Path
from env.Include.lib.functions import *
from env.Include.ml.imports import *

def savePlt(plt, OutDir, FileName):
  setOrCreatePath(OutDir)
  plt.savefig(OutDir + FileName)
  print("savePlt: " + OutDir + FileName)
  plt.close()

def show2dScatter(train_X, train_y, y, x, regressor, thisModelName, show):
  if show in ['inline', 'file']:
    graphCongInit()
    # Visualising 
    plt.scatter(train_X, train_y, color='red')
    plt.plot(train_X, regressor.predict(train_X), color='blue')
    plt.title(str("'{0}' vs '{1}' SLR prediction (training set)".format(y, x)))
    plt.xlabel("{0}".format(x))
    plt.ylabel("{0}".format(y))
    if show == 'inline':
      plt.show() 
    elif show == 'file':
      # SET WRITE DIRECTORY
      outDir = "results/ML/" + thisModelName
      setOrCreatePath(outDir)
      plt.savefig(outDir + "/" + x + "_vs_" + y + '.png', bbox_inches='tight')
      print(outDir + "/" + x + "_vs_" + y + ".png Created")
  else:
    print("Missconfigured show")
  plt.close()

def showCorrHeatMap(df, thisModelName, x, y, show):
  if show in ['inline', 'file']:
    _ , ax = plt.subplots(figsize =(20, 16))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    if df.shape[1] <= 5:
      fntSize = 18
    elif df.shape[1] < 10:
      fntSize = 14
    else:
      fntSize = 12
    
    sns.set(font_scale=1.4)
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':fntSize }
    )
  plt.title('Pearson Correlation of Features', y=1.05, size=15)
  if show == 'inline':
      plt.show() 
  elif show == 'file':
    # SET WRITE DIRECTORY
    outDir = "results/ML/" + thisModelName
    setOrCreatePath(outDir)
    plt.savefig(outDir + "/" + x + "_vs_" + y + '.png', bbox_inches='tight')
    print(outDir + "/" + x + "_vs_" + y + ".png Created")
  else:
    print("Missconfigured show")
  plt.close()