'''Different vector index types with add, search, delete operations'''
import os, json
import numpy as np
from shortuuid import uuid
from random import randint
from utils import *

class VanillaIndex :
    def __init__(self, index_name:str, index_dim=512, index_type='IVFFlat', index_path=f"/home/suppra/Desktop/pinecone/Indexes") :
        self.index_name = index_name
        self.index_dim = index_dim
        self.index_path = f'index_path/{index_type}_{index_name}'
        self.vectorType = [('_id', 'i4'), ('vector', int, (4,)), ('metadata', dict)]
            
        # Create new index if it doesn't exist
        if not os.path.isdir(self.index_path) :
            os.mkdir(self.index_path)
        
            # Creating a descriptor for the index
            self.descriptor = {"Name" : self.index_name, 
                               "Dimensions" : self.index_dim, 
                               "Namespaces" : {}}       
        else :
            self.descriptor = json.loads(open(f'{self.index_path}/descriptor.json', 'a').read())
    
    def getNamespace(self, namespace='default') :
        '''Check if a namespace exists, else create one and return its path'''
        namespacePath = f'{self.index_path}/{namespace}'
        if not os.path.isdir(namespacePath) :
            os.mkdir(namespacePath)
            self.descriptor['Namespaces'][namespace] = {'Vectors' : 0, 'Centroid' : ''}
        return namespacePath

    def upsert(self, vector, namespace='default') :
        '''Upsert a single vector into the namespace'''
        namespacePath = self.getNamespace(namespace=namespace)
        try :
            vecID = uuid()
            np.save(f'{namespacePath}/uuid.npy', vector)
            self.descriptor['Namespaces'][namespace]['Vectors'] += 1
            return 1
        except Exception as e :
            print(str(e))
            return 0

    def upsert_batch(self, vectors:list, namespace='default') :
        '''Upsert multiple vectors into the namespace'''
        for vec in vectors :
            res = self.upsert(vector=vec, namespace=namespace)
            if not res :
                return res
            
    def flushIndex(self) :
        '''Run when you're done operating with the index.'''
        with open(f'{self.index_path}/descriptor.json', 'a') as desc :
            desc.write(json.dumps(self.descriptor))
    

class IVFFlat(VanillaIndex) :
    def __init__(self, index_name:str) :
        self.index_type = 'IVFFlat'
        super.__init__(index_name=index_name, index_type = self.index_type)

    def cluster(self, namespace='default', min_threshold=30) :
        '''Create clusters only when there are mininum number of vectors in a namespace
            Called after upsert operation'''
        namespacePath = f'{self.index_path}/{namespace}'
        lenVectors = self.descriptor['Namespaces'][namespace]['Vectors']
        if lenVectors >= min_threshold :
            self.KMeansClustering(namespacePath, k=4)
        else : 
            return