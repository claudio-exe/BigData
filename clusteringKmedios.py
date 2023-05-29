from pyclustering.cluster import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import csv
import pandas as pd

# sample data points
df = pd.read_csv = ''
valutazione = df['valutazione']
prezzo = df['prezzo']


# number of clusters
kmeansVal = KMeans(n_clusters=2)
kmeansPrez = KMeans(n_clusters=2)

# fitting the k means algorithm on scaled data
kmeansVal.fit(valutazione)
kmeansPrez.fit(prezzo)

# predict the clusters
predictionValutazione = kmeansVal.predict(valutazione)
predictionPrezzo = kmeansPrez.predict(prezzo)

# centroids values
centroidsVal = kmeansVal.cluster_centers_
centroidsPrez = kmeansPrez.cluster_centers_

# print results
print("Cluster valutazione: ",predictionValutazione)
print("Centroids valutazione: ", centroidsVal)
print("Cluster prezzo: ",predictionPrezzo)
print("Centroids prezzo: ",centroidsPrez)

# grafico cluster valutazione
plt.scatter(valutazione[:, 0], valutazione[:, 1], c=predictionValutazione)
plt.scatter(centroidsVal[:, 0], centroidsVal[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.show()
# grafico cluster prezzo
plt.scatter(prezzo[:, 0], prezzo[:, 1], c=predictionPrezzo)
plt.scatter(centroidsPrez[:, 0], centroidsPrez[:, 1], marker='x', s=200, linewidths=3, color='r')
plt.show()




#sample = np.array([[1.2, 2.2], [1.2, 4.5], [1.7, 0.3], [4.3, 2.3], [4.9, 4.9], [4.5, 0.6], [3.2,1.8],[3.2,2.8]])
#kmeans = KMeans(n_clusters=2)
#kmeans.fit(sample)
#predictions = kmeans.predict(sample)
#centroids = kmeans.cluster_centers_
#print("Cluster predictions:", predictions)
#print("Centroids:", centroids)
#plt.scatter(sample[:, 0], sample[:, 1], c=predictions)
#plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='r')
#plt.show()
