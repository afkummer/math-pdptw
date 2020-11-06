#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

from sys import argv

import random
import numpy as np
import matplotlib.pyplot as plt
import csv 		

from scipy.cluster.vq import kmeans, vq, whiten

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, robust_scale, minmax_scale, maxabs_scale

def bench_k_means(estimator, name, data):
   """ Runs the benchmarking of the k-means algorithm. """
   estimator.fit(data)
   print("name: %s   inertia_: %f   silhouette_score: %f" % (name, estimator.inertia_, metrics.silhouette_score(data, estimator.labels_)))

def attach_mouse_listener(fig, sc, names):
   """ Allows visualizing the data associated with the points. """
   # Prepare the anottation that receives the data.
   annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
      bbox=dict(boxstyle="round", fc="w"),
      arrowprops=dict(arrowstyle="->"))
   annot.set_visible(False)
   
   # Defines the callback for mouse motion.
   def hover(event):
      vis = annot.get_visible()
      if event.inaxes == ax:
         cont, ind = sc.contains(event)
         if cont:
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            annot.zorder = 10
            text = names[ind["ind"][0]]
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.8)
            annot.set_visible(True)
         else:
            if vis: annot.set_visible(False)
         fig.canvas.draw_idle()

   # Attach the callback to the plot.
   fig.canvas.mpl_connect("motion_notify_event", hover)


if __name__ == "__main__":
   if len(argv) != 4:
      print("Usage: %s <1:seed> <2:features csv> <3:num clusters>" % argv[0])
      exit(1)

   # Parse command line and print the script arguments.   
   print("--- Clustering with k-mens ---")
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

   # Runs a quick benchmark with several cluster variations.
   print('-' * 80)
   print("Benchmarking with several k values: ")
   rang = 10
   for k in range(max(2, num_clusters-rang), min(len(instance_names)-1, num_clusters+rang)):
      bench_k_means(KMeans(init='k-means++', n_clusters=k, n_init=10),
         name="k-means++ (k=" + str(k) + ")", data=features_data)
   print('-' * 80)

   # Prepare the data for visualization in 2D-plot.
   reduced_data = PCA(n_components=2, random_state=np.random.randint(1)).fit_transform(features_data)
   kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
   kmeans.fit(reduced_data)

   # Step size of the mesh. Decrease to increase the quality of the VQ.
   h = .005     # point in the mesh [x_min, x_max]x[y_min, y_max].

   # Plot the decision boundary. For that, we will assign a color to each
   x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
   y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
   xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

   # Obtain labels for each point in mesh. Use last trained model.
   Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

   # Put the result into a color plot
   Z = Z.reshape(xx.shape)
   fig, ax = plt.subplots()
   plt.imshow(Z, interpolation='nearest',
      extent=(xx.min(), xx.max(), yy.min(), yy.max()),
      cmap=plt.cm.Paired,
      aspect='auto', origin='lower')
   
   # Plot the centroids as a white x.
   centroids = kmeans.cluster_centers_
   plt.scatter(centroids[:, 0], centroids[:, 1],
               marker='x', s=169, linewidths=3,
               color='w', zorder=4)

   # Plot the instance points.
   sc = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='k', s=3)

   # Define the limits of the plotting area.
   plt.title('K-means clustering (PCA-reduced data)\nCentroids are marked with white cross')
   plt.xlim(x_min, x_max)
   plt.ylim(y_min, y_max)

   # Format the names and the distances to cluster centroids of the instances.
   idx, dist = vq(reduced_data, centroids)
   text_hover = []
   for i in range(features_data.shape[0]):
      txt = "%s (d=%.3f)" % (instance_names[i], dist[i])
      text_hover.append(txt)

   # Attach the event listener for capturing the mouse moves.
   attach_mouse_listener(fig, sc, text_hover)

   
   # Write the clustering to a csv file. Also outputs to the stdout.
   print("Clustering outputs:")
   with open("clustering-" + str(seed) + "-" + str(num_clusters) + ".csv", "w") as fid:
      fid.write("instance,cluster,dist.centroid\n")
      names = np.vstack(instance_names)
      for i in range(num_clusters):
         print("-" * 80)
         print("Cluster: %d" % i)
         rn = names[idx==i, 0]
         rd = dist[idx==i]
         for j in range(len(rd)):
            print("%s\t%.4f" % (rn[j], rd[j]))
            fid.write("%s,%d,%f\n" % (rn[j], i, rd[j]))       

   # Save the plot into a PDF file, then show it in the screen.
   plt.savefig("clustering-" + str(seed) + "-" + str(num_clusters) + ".pdf")
   plt.show()
