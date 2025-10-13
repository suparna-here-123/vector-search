import os
from math import sqrt
from random import sample
import numpy as np

def KMeansClustering(path: str, k: int):
    '''Creates k clusters for a given directory(namespace/cluster)'''
    # 1) Bring vectors in path into memory
    vectors = []
    for file in os.listdir(path) :
        vectors.append(np.load(os.path.join(path, file), allow_pickle=True))

    # 2) Choose k random vectors to be cluster centers - IMPROVISE
    centers = sample(vectors, 3)
    
    # 3) Assign vectors to centers
    clusters = getClusters(vectors, centers)

    # 4) Update centers for N times
    updateCenters(clusters)

    # 5) Return clusters
    return clusters

def getClusters(vectors, centers) :
        print(f"Inside getClusters - received {len(vectors)} vectors")
        # Create a cluster with center:vector key-value pair
        numCenters = len(centers)
        
        # Initialize the cluster
        clusters = [{'center' : centers[i], 'vectors' : []} for i in range(numCenters)]
        
        # Assign vectors to closest center
        for i, vector in enumerate(vectors) :
            justVector = vector[0][1]                 
            distFromcenters = [0] * numCenters

            # Calculate distance of this vector from all centers
            for centerIndex in range(numCenters):
                justCenter = centers[centerIndex][0][1]
                dis = euclideanDistance(justCenter, justVector)
                distFromcenters[centerIndex] = dis
            
            toCluster = np.argmin(distFromcenters)
            clusters[toCluster]['vectors'].append(vector)

        return clusters

def updateCenters(clusters) :
    for cluster in clusters :
        justVectors = np.array([vec[0][1] for vec in cluster['vectors']])
        new_center = justVectors.mean(axis=0)

        # Cluster center will not have any metadata - just an ndarray
        cluster['center'] = new_center
        print(new_center)

def euclideanDistance(v1, v2) :
    return np.sqrt(np.sum((v1 - v2) ** 2))

def nearestCluster(v, clusters:dict) :
    '''Return the path and distance of the nearest cluster whose centroid is closest to given query vector'''
    cPaths = []
    distances = []
    # Iterating over every cluster's centroid and finding query vector distance from it
    for c, cPath in clusters.items() :
        cPaths.append(cPath)
        centroid = np.load(f'{cPath}/center.py', allow_pickle=True)
        dist = euclideanDistance(v[0][1], centroid[0][1])
        distances.append(dist)
    
    # Return path of the cluster whose centroid is closest to query vector
    closestClusterIndex = np.argmin(distances)
    return [cPaths[closestClusterIndex], distances[closestClusterIndex]]
