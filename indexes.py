'''Different vector index types with add, search, delete operations'''
import os, json
import numpy as np
from shortuuid import uuid
from random import randint
from utils import *
from abc import ABC, abstractmethod

class VanillaIndex(ABC) :
    def __init__(self, index_name:str, index_dim=512, index_type=str, index_path=f"/home/suppra/Desktop/pinecone/Indexes") :
        self.index_name = index_name
        self.index_dim = index_dim
        self.index_path = f'{index_path}/{index_type}_{index_name}'
        self.vectorType = [('_id', 'i4'), ('vector', int, (4,)), ('metadata', dict)]
            
        # Create new index if it doesn't exist
        if not os.path.isdir(self.index_path) :
            os.mkdir(self.index_path)
        
            # Creating a descriptor for the index
            self.descriptor = {"Name" : self.index_name, 
                               "Dimensions" : self.index_dim, 
                               "Namespaces" : {}}
            self.flushIndex()    
        else :
            self.descriptor = json.loads(open(f'{self.index_path}/descriptor.json', 'r').read())
    
    def getNamespace(self, namespace='default') :
        '''Return path of a namespace, if it's non-existent, create one.'''
        namespacePath = f'{self.index_path}/{namespace}'
        if not os.path.isdir(namespacePath) :
            os.mkdir(namespacePath)
            self.descriptor['Namespaces'][namespace] = {'Vectors' : 0, 'Clusters' : {}}
            self.flushIndex()
        return namespacePath

    def upsert(self, vector, namespace='default') :
        '''Upsert a single vector into the namespace after clusters have been created'''
        namespacePath = self.getNamespace(namespace=namespace)
        try :
            vecID = uuid()

            # Should assign it to the right cluster
            # 1) Extract all the cluster centers from storage
            clusterPaths = []
            for item in os.listdir(namespacePath) :
                clusterPath = os.path.join(namespacePath, item)
                clusterPaths.append(clusterPath)
            

            # 2) Find nearest cluster
            closestClusterPath, distance = nearestCluster(vector , clusterPaths)
            print(f'{vector} is closest to {closestClusterPath} at {distance}')

            # 3) Save vector in that cluster
            np.save(f'{closestClusterPath}/{vecID}.npy', vector)
            print(f"Saved vector to {closestClusterPath}")

            # 4) Increase count of vectors in namespace by 1
            self.descriptor['Namespaces'][namespace]['Vectors'] += 1
            self.flushIndex()

            return 1
        
        except Exception as e :
            print(str(e))
            return 0

    def upsert_batch(self, vectors:list, namespace='default') :
        '''Upsert multiple vectors into the namespace'''
        namespacePath = self.getNamespace(namespace=namespace)
        for vec in vectors :
            # Create a unique name for this vector            
            vecID = uuid()

            # Save vector into namespace
            np.save(f'{namespacePath}/{vecID}.npy', vec)

            # Increase count of vectors in namespace by 1
            self.descriptor['Namespaces'][namespace]['Vectors'] += 1
            
        self.flushIndex()

        # Check if threshold for clustering has been reached, if so, organise
        self.organise(namespace)
        return 1

    @abstractmethod
    def organise(self, namespace:str, min_threshold:int) :
        '''Organisation of vectors in a namespace - dependent on type of index'''
        pass
    
    @abstractmethod
    def search(self, namespace:str, query, topK:int) :
        '''Given a query vector, return the top K most similar vectors along w cosine similarity'''
        pass

    def flushIndex(self) :
        '''Run when description needs to be written to disk'''
        with open(f'{self.index_path}/descriptor.json', 'w') as desc :
            desc.write(json.dumps(self.descriptor, indent=4))
    
    def saveClusters(self, namespace:str, namespacePath:str, clusters:list[dict]) :
        try :
            for clusterInd, cluster in enumerate(clusters) :
                # Create new directory for cluster if it doesn't exist
                clusterPath = f'{namespacePath}/cluster_{clusterInd}'
                if not os.path.isdir(clusterPath) :
                    os.mkdir(clusterPath)
                    print(f"Made path {clusterPath}")
                
                # Save center
                np.save(f'{clusterPath}/center.npy', cluster['center'])
                self.descriptor['Namespaces'][namespace]['Clusters'].update({f'C{clusterInd}' : clusterPath})
                self.flushIndex()

                # Save vectors
                for vec in cluster['vectors'] :
                    np.save(f'{clusterPath}/{uuid()}.npy', vec)
            
            # Delete the now duplicated vectors under the namespace
            for item in os.listdir(namespacePath) :
                item_path = os.path.join(namespacePath, item)
                if os.path.isfile(item_path) :
                    os.remove(item_path)
            
            # Return success code
            return 1
    
        except Exception as e :
            print(f'The error is : {e}')
            return 0          
            

class IVFFlat(VanillaIndex) :
    def __init__(self, index_name:str) :
        self.index_type = 'IVFFlat'
        super().__init__(index_name=index_name, index_type = self.index_type)

    def organise(self, namespace='default', min_threshold=8) :
        '''
        Create clusters only when there are mininum number of vectors in a namespace.
        Called after upsert operation.
        '''
        lenVectors = self.descriptor['Namespaces'][namespace]['Vectors']
        if lenVectors >= min_threshold :
            # If first-time clustering...
            if not self.descriptor['Namespaces'][namespace].get('Clusters', None) :
                namespacePath = f'{self.index_path}/{namespace}'
                clusters = KMeansClustering(namespacePath, k=4)
                if self.saveClusters(namespace, namespacePath, clusters) :
                    return 1

            # Clusters already exist :
            else :
                print("Do something")
        
        else : 
            return 0
    
    def search(self, query, namespace='default', topK=3) :
        # Compare vector with all centroids in this namespace and find nearest one
        Q = query[0][1]
        nsInfo = self.descriptor['Namespaces'][namespace]
        cluster_name_path = nsInfo.get('Clusters', None)

        # By default, search the whole namespace if there are no clusters yet
        searchPath = self.getNamespace(namespace)

        # If there are clusters, search in closest one
        if cluster_name_path :
            clusterPaths = list(cluster_name_path.values())
            nearestClusterPath, distance = nearestCluster(query, clusterPaths)
            print(f"Nearest cluster = {nearestClusterPath} at distance {distance}")
            searchPath = nearestClusterPath
        
        print(f"Searching in {searchPath}")
        # Do brute-force search/sort
        vec_dist = []
        for vecPath in os.listdir(searchPath) :
            if not vecPath.startswith('center') :
                vec = np.load(os.path.join(searchPath, vecPath), allow_pickle=True)
                vec_dist.append((vec, euclideanDistance(Q, vec[0][1])))
        
        # Sort vectors by distance from query
        vec_dist.sort(key=lambda item : item[1])

        print("Distances")
        for i in vec_dist :
            print(i)

        # Return top K vectors with their distances from query
        return vec_dist[:topK]

if __name__ == "__main__" :
    custom = [('_id', 'i4'), ('vector', int, (4,)), ('metadata', dict)]

    # a1 = np.array([1,2,3,4])
    # a2 = np.array([1,2,5,4])
    # a3 = np.array([1,12,3,4])
    # a4 = np.array([1,12,3,6])
    # a5 = np.array([1,2,8,4])
    # a6 = np.array([1,2,8,5])
    # a7 = np.array([0,12,2,4])
    # a8 = np.array([0,11,3,6])

    # v1 = np.array([(1, a1, {'docId' : 15, 'type' : 'political'})], dtype=custom)
    # v2 = np.array([(2, a2, {'docId' : 13, 'type' : 'political'})], dtype=custom)
    # v3 = np.array([(3, a3, {'docId' : 15, 'type' : 'economics'})], dtype=custom)
    # v4 = np.array([(4, a4, {'docId' : 15, 'type' : 'economics'})], dtype=custom)
    # v5 = np.array([(5, a5, {'docId' : 14, 'type' : 'biology'})], dtype=custom)
    # v6 = np.array([(6, a6, {'docId' : 12, 'type' : 'biology'})], dtype=custom)
    # v7 = np.array([(7, a7, {'docId' : 12, 'type' : 'history'})], dtype=custom)
    # v8 = np.array([(8, a8, {'docId' : 15, 'type' : 'history'})], dtype=custom)

    idx = IVFFlat('ABC')
    # resUpsert = idx.upsert_batch([v1, v2, v3, v4, v5, v6, v7, v8], namespace='users')
    # print('Batch upsert operation success : ', bool(resUpsert))

    # a9 = np.array([1,12,3,5])
    # v9 = np.array([(9, a9, {'docId' : 15, 'type' : 'economics'})], dtype=custom)
    # idx.upsert(v9, namespace='users')

    # a10 = np.array([1,12,3,7])
    # v10 = np.array([(10, a10, {'docId' : 15, 'type' : 'economics'})], dtype=custom)
    # idx.upsert(v10, namespace='users')

    a = np.array([1,2,3,4])
    v = np.array([(11, a, {'docId' : 15, 'type' : 'biology'})], dtype=custom)
    topK = idx.search(v, 'users')
    for i in topK :
        print(i)