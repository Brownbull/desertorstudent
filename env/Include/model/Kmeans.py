# -*- coding: utf-8 -*-
"""
@author: Brownbull - Gabriel Carcamo - carcamo.gabriel@gmail.com
  K Means Clustering
"""
from env.Include.model.imports import *
from env.Include.model.tools import *
from env.Include.model.processing import *
from env.Include.model.visual import *
from sklearn.cluster import KMeans

def KMeans_elbow(outDir, modelName, dataset, config):
  # Using elbow method to find optimal number of custers
  X = dataset[[config['x'], config['y']]]
  wcss = []
  for i in range(1,11):
      kmeans = KMeans(
              n_clusters = i, 
              init = 'k-means++', 
              max_iter = 300, 
              n_init = 10, 
              random_state = 0)
      kmeans.fit(X)
      wcss.append(kmeans.inertia_)

  plt.cla()
  plt.plot(range(1,11), wcss)
  plt.title(modelName + ': Elbow Method')
  plt.xlabel('Number of CLusters')
  plt.ylabel('WCSS')
  savePlt(plt, outDir, "Elbow.png")

  # Calculate distance between points
  wcssDiffs = []
  wcssPrev = 0
  for e in wcss:
    if wcssPrev != 0:
      wcssDiffs.append(wcssPrev - e)
    wcssPrev = e

  # Calculate % of information gain
  wcssPercs = []
  wcssPrev = 0
  for e in wcssDiffs:
    if wcssPrev != 0:
      wcssPercs.append(1 - (e/wcssPrev))
    wcssPrev = e

  # Get number of clusters (Where significant information gain below 50%)
  n_clusters = 1
  for i, e in enumerate(wcssPercs):
    if( e < 0.5000):
      return  i + 1
  return n_clusters
  
def KMeans_train(outDir, modelName, dataset, config, n_clusters):
  # Select Features
  X = pd.DataFrame()
  X[config['x']] = dataset[config['x']].astype(np.int64)
  # Select Target 
  X[config['y']] = dataset[config['y']].astype(np.int64)
  # Transform to Array
  X = X.values
 
  kmeans = KMeans(
              n_clusters = n_clusters, 
              init = 'k-means++', 
              max_iter = 300, 
              n_init = 10, 
              random_state = 0)
  # kmeans.fit(X)
  y_kmeans = kmeans.fit_predict(X)
  print("y_kmeans: " + str(len(y_kmeans)))

  # Visualize 2D only
  plt.cla()
  plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 30, color = 'red', label = 'C1')
  plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 30, color = 'blue', label = 'C2')
  plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 30, color = 'green', label = 'C3')
  # plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, color = 'cyan', label = 'Careless')
  # plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, color = 'magenta', label = 'Sensible')
  plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
  plt.title('KNN Means ' + modelName)
  plt.xlabel(config['x'])
  plt.ylabel(config['y'])
  savePlt(plt, outDir, str(n_clusters) + "Clusters.png")

  return {
    'config': config,
    'model': kmeans,
    'x' : config['x'], 
    'y' : config['y'],
    'test_y' : None,
    'pred_y': None
  }
