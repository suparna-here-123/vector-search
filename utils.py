import os
from math import sqrt
from random import randint
import numpy as np

def KMeansClustering(self, path:str, k:int) :
    '''Creates k clusters for a given directory(namespace/cluster)'''
    # 1) Bring vectors in path into memory
    vectors = []
    for file in os.path.listdir(path) :
        vectors.append(np.load(file, allow_pickle=True))
    
    # 2) Choose k random vectors to be cluster centroids - IMPROVISE
    centroidIndexes = randint(0, len(vectors)-1)
    centroids = []
    for centroidIndex in centroidIndexes :
        centroids.append(vectors.pop(centroidIndex))
    
    # 3) Assign vectors to centroids

def assignToCluster(self, vectors, centroids) :
        numCentroids = len(centroids)
        for vec in vectors :
            distFromCentroids = [0] * numCentroids

            # Calculate distance of this vector from all centroids
            for centroidIndex in range(numCentroids):
                dis = euclideanDistance(centroids[centroidIndex][0], vec)
                distFromCentroids[centroidIndex] = dis
            
            curr_cluster = np.argmin(distFromCentroids)
            

def euclideanDistance(vec1, vec2) :
    v1, v2 = vec1[1], vec2[1]
    return np.sqrt(np.sum((v1 - v2) ** 2))