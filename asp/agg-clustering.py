#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

from sys import argv

import random
from time import time
import numpy as np
import matplotlib.pyplot as plt
import csv 		

from scipy.cluster.vq import kmeans, vq, whiten

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale
from sklearn.cluster import AgglomerativeClustering
from sklearn import manifold

if __name__ == "__main__":
   if len(argv) != 4:
      print("Usage: %s <1:seed> <2:features csv> <3:num clusters>" % argv[0])
      exit(1)

   # Parse command line and print the script arguments.   
   print("--- AgglomerativeClustering ---")
   seed = int(argv[1])
   print("Seed: " + str(seed))
   feat_csv = argv[2]
   print("Features CSV: " + feat_csv)
   num_clusters = int(argv[3])
   print("Number of clusters to generate: " + str(num_clusters))

   # Seeding of the RNG.
   # SciKit-learn uses only the np.random, but I prefer to play safe
   # and seed also the random from stdlib.
   np.random.seed(seed)
   random.seed(seed)

   print()

   # Importing of the features CSV.
   with open(feat_csv) as fid:
      reader = csv.reader(fid)
      feature_names = [x.strip() for x in fid.readline().split(",")[1:]]
      instance_names = []
      features_data = []
      for row in reader:
         instance_names.append(row[0].strip())
         features_data.append([float(x) for x in row[1:]])

   # Convert the data to the format used internally in numpy and scipy.
   features_data = np.vstack(features_data)
   print("The input data contains %d instances and %d features." % (features_data.shape[0], features_data.shape[1]))
   print("Possible number of clusters to generate: 1 up to %d." % features_data.shape[1])

   print()

   # Normalize the input data. 
   features_data = scale(features_data)
   # X_red = features_data 
   # X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(features_data)
   X_red = PCA(n_components=2, random_state=np.random.randint(1)).fit_transform(features_data)

   #----------------------------------------------------------------------
   # Visualize the clustering
   def plot_clustering(X_red, labels, title=None):
      x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
      X_red = (X_red - x_min) / (x_max - x_min)

      # plt.figure(figsize=(6, 4))
      plt.figure()
      for i in range(X_red.shape[0]):
         plt.plot(X_red[i, 0], X_red[i, 1], marker="o", markersize=3,
                  color=plt.cm.nipy_spectral(labels[i] / num_clusters))

      if title is not None:
         plt.title(title)
      # plt.axis('off')
      # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
   
   # for linkage in ['ward']: #('ward', 'average', 'complete', 'single'):
   for linkage in ('ward', 'average', 'complete', 'single'):
      # clustering = AgglomerativeClustering(linkage=linkage, n_clusters=num_clusters)      
      clustering = AgglomerativeClustering(linkage=linkage, n_clusters=None, distance_threshold=4.0)
      t0 = time()
      clustering.fit(X_red)
      num_clusters = clustering.n_clusters_
      print("%s :\t%.2fs" % (linkage, time() - t0))

      plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)
      print("Clusters identified (%s): %d" % (linkage, clustering.n_clusters_))
   
   plt.show()
