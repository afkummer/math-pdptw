#https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# This code builds a decision tree according to the input data
# The trained data input should be a file in csv format with the first line containing the name of columns, the first column containing classes, and the second column containing the name of each instance
# The pred data (instances to predict which class belong to) input should be a file in csv format with the first line containing the name of columns and the second column containing the name of each instance
#automaticaly a pdf file is saved with the decison tree
#clusterID.csv is the cluster file returned by the clustering algorithm
# USAGE: phyton3 decision-tree.py seed train-data_file.csv predic-data_file.csv clusterID.csv
# EXAMPLE: python3 decision-tree.py 31415 all-features1802.csp pred-all-instances-features-1802.csv clustID-8-31415.csv 

from sklearn import tree
from graphviz import Source
from sklearn.tree import export_text


import matplotlib.pyplot as plt #para plotar a arvore

import numpy as np
import csv
import copy
import sys              #for reading command line arguments
from numpy            import vstack,array
import graphviz  #for ploting the decision tree
decisionT = tree.DecisionTreeClassifier(random_state=0)

np.random.seed(int(sys.argv[1])) #set a seed to always generate the same cluster

#bloco
data_arr = []
instance_name_arr = []
instance_class_arr = []
instF = {} #dictionary

#read the classes
with open(sys.argv[4], newline='') as f:
     reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
     for row in reader:
        instF[row[0].strip()]=row[1]

#training data
with open(sys.argv[2], newline='') as f:
     reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
     #remove the features line and save it in features_arr list
     feature_arr = next(reader)
     for row in reader:
        data_arr.append([float(x) for x in row[1:]])
        instance_name_arr.append(row[0])
        clusterID=instF[row[0].strip()]
        instance_class_arr.append(clusterID)

feature =  vstack (feature_arr)
feature = np.delete(feature,0) #remove name of the first column = name of the instances
#print(feature)
#print(len(feature))

#prediction-------
data2_arr = []
pred_instance_name_arr = []
feature_arr = []

with open(sys.argv[3], newline='') as f:
     reader2 = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
     next(reader2) #remove first line
     for row in reader2:
         data2_arr.append([float(x) for x in row[1:]])
         pred_instance_name_arr.append([row[0].strip()])


pred_instance_name = vstack(pred_instance_name_arr)
data2 = vstack( data2_arr )
#---------------


data = vstack( data_arr )
instance_name = vstack(instance_name_arr)
instance_class = vstack(instance_class_arr)
#feature = vstack(row0)
#remove the first two itens
#feature = np.delete(feature,1)
#feature = np.delete(feature,0)


#print(instance_name)
#print(instance_class)

#Create a decision tree classifier object
decisionT = tree.DecisionTreeClassifier()
#Train and plot decision tree classifier
decisionT = decisionT.fit(data, instance_class)


#plotting the decision tree
tree.plot_tree(decisionT,filled=True)
#tree.plot_tree(decisionT)

#------------------------------
#fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
#cn=['setosa', 'versicolor', 'virginica']

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(decisionT,feature_names = feature,filled=True)
#tree.plot_tree(decisionT,
#               feature_names = feature, 
#               class_names=instance_class)
#               filled = True);
filename = "DecisionT" + "-" + sys.argv[1] + sys.argv[4][7:-4] + ".png"
fig.savefig(filename)

#------------------------------

#Measuring Model Performance
score = decisionT.score(data, instance_class)
print(score)

#predict hte response for data2
y_pred = decisionT.predict(data2)
# print the predicted price 
print(y_pred) 
print(pred_instance_name)

# Print a textual decision tree: Build a text report showing the rules of a decision tree
#r = export_text(decisionT, feature_names=feature)
#print(r)
#----------------

#https://chrisalbon.com/machine_learning/trees_and_forests/visualize_a_decision_tree/
#dot_data = tree.export_graphviz(decisionT, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("MIRA") 
#dot_data = tree.export_graphviz(decisionT, out_file=None,feature_names=feature,class_names=instance_class,filled=True,rounded=True,special_characters=True)  
#graph = graphviz.Source(dot_data)  
#graph

