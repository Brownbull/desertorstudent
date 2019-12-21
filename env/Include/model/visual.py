from pathlib import Path
from env.Include.model.tools import *
from env.Include.model.imports import *

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

def showCorrHeatMap(df, thisModelName, x, y, show):
  if show in ['inline', 'file']:
    _ , ax = plt.subplots(figsize =(20, 16))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
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