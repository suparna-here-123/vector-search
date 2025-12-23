### VectorSoup - A tiny implementation of vector indexes

#### Vanilla Index methods explained
1. init: Initializes index name, dimensions of vectors in it, and vector type. Creates a new index (folder in this implementation). Add metadata to an index descriptor.json file.

2. getNamespace: Return path of a namespace if it exists, else create one.

3. upsert: Add a single vector into a namespace (and nearest cluster) given that initial clusters have been created.

4. upsert_batch: Argument takes multiple vectors to be upserted. Calls _organise()_ to create clusters if minimum threshold reached.

5. organise: Implementation depends on type of index, for IVFFlat, if minimum number of vectors in the namespace exist, create 4 clusters using K-means clustering.

6. search: Implementation depends on type of index, for IVFFlat, find nearest cluster centroid for query vector, and do brute-force search within that cluster, return top 3 nearest vectors. 