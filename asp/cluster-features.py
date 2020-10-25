# This code forms k clusters with the instances given as input.
# the number of clusters k is given as input
# The data input should be a file in csv format with the first line containing the name of columns and the first column containing the name of each instance
# USAGE: phyton3 cluster-features.py K seed data_file.csv
# it generates a CSV file with the clusters
# EXAMPLE: python3 cluster-features.py 8 31415 features-all-instances.csv
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from numpy            import vstack,array
from numpy.random     import rand
from scipy.cluster.vq import kmeans, vq, whiten
import csv 		#for reading csv file
import sys 		#for reading command line arguments


from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(int(sys.argv[2])) #set a seed to always generate the same cluster


#n_digits defines the number of clusters
n_digits=int(sys.argv[1])


#bloco
data_arr = []
instance_name_arr = []

with open(sys.argv[3], newline='') as f:
     reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
     next(reader) #remove the first line of the csv file
     for row in reader:
         data_arr.append([float(x) for x in row[1:]])
         instance_name_arr.append([row[0]])

data = vstack( data_arr )
instance_name = vstack(instance_name_arr)

# normalization
#data = whiten(data)

#Standardization, or mean removal and variance scaling
data = scale(data_arr)

n_samples, n_features = data.shape

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
#    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
#          % (name, (time() - t0), estimator.inertia_,
#             metrics.homogeneity_score(labels, estimator.labels_),
#             metrics.completeness_score(labels, estimator.labels_),
#             metrics.v_measure_score(labels, estimator.labels_),
#             metrics.adjusted_rand_score(labels, estimator.labels_),
#             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
#             metrics.silhouette_score(data, estimator.labels_,
#                                      metric='euclidean',
#                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=4),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=4),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')


# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=4)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=4)
plt.title('K-means clustering (PCA-reduced data)\n'
          'Centroids are marked with white cross')

 # assign each sample to a cluster
idx,_ = vq(reduced_data,centroids)


# print results and save result in a file
# Open the file with writing permission

filename = "clustID" + "-" + sys.argv[1] + "-" + sys.argv[2] + ".csv"
textfile = open(filename, 'w')



for i in range(n_digits):
    result_names = instance_name[idx==i, 0]
    print("=================================")
    print("Cluster " + str(i+1))
    for name in result_names:
        print(name.strip())
        s=','
        #strip removes the white spaces of the string - there was one at the end
        textfile.write(name.strip())
        textfile.write(s)
        textfile.write('{:d}\n'.format(i))

textfile.close()


plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
