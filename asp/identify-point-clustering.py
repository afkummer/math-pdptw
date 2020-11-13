#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

from time import time
import random

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
import csv

if __name__ == "__main__":
   from sys import argv
   if len(argv) != 4:
      print("Usage: %s <1:seed> <2:points csv> <3:dist treshold>" % argv[0])
      exit(1)

   # Parse command line and print the script arguments.   
   print("--- PD region detector with AgglomerativeClustering ---")
   seed = int(argv[1])
   print("Seed: " + str(seed))
   points_csv = argv[2]
   print("P&D locations CSV: " + points_csv)
   dist_threshold = float(argv[3])
   print("Distance thresold: " + str(dist_threshold))

   # Seeding of the RNG.
   # SciKit-learn uses only the np.random, but I prefer to play safe
   # and seed also the random from stdlib.
   np.random.seed(seed)
   random.seed(seed)

   print()

   # Importing of the features CSV.
   with open(points_csv) as fid:
      reader = csv.reader(fid)
      fid.readline() # Discards the headers.
      point_data = []
      for row in reader:
         point_data.append([float(x) for x in row[1:]])

   # Convert the data to the format used internally in numpy and scipy.
   point_data = np.vstack(point_data)
   print("The input data contains %d points." % point_data.shape[0])

   print()      

   X = point_data
   # X, y = datasets.load_digits(return_X_y=True)
   # n_samples, n_features = X.shape

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
         plt.title(title, size=17)
      # plt.axis('off')
      # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
   
   from sklearn.preprocessing import scale
   X_red = scale(X)
   # X_red = X

   # for linkage in ['ward']: #('ward', 'average', 'complete', 'single'):
   for linkage in ('ward', 'average', 'complete', 'single'):
      # clustering = AgglomerativeClustering(linkage=linkage, n_clusters=num_clusters)
      clustering = AgglomerativeClustering(linkage=linkage, n_clusters=None, distance_threshold=dist_threshold)
      t0 = time()
      clustering.fit(X_red)
      print("%s :\t%.2fs" % (linkage, time() - t0))

      num_clusters = clustering.n_clusters_
      plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)
      print("Clusters identified (%s): %d" % (linkage, clustering.n_clusters_))

   
   plt.show()
