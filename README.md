### VectorSoup - A tiny implementation of vector indexes

I wanted to understand how different vector indexes and databases work, given that they are the backbone of any AI application today.
With the help of [this medium article](https://medium.com/@myscale/understanding-vector-indexing-a-comprehensive-guide-d1abe36ccd3c), I have implemented the first, most basic vector index called IVFFlat (Inverted File Flat).
While I am still in the process of implementing other more complex types, this project gave me an opportunity to use and understand the importance object-oriented programming principles and to appreciate the math behind vectors and leveraging them for better query-search results.


#### Vanilla Index methods explained
1. _init_: Initializes index name, dimensions of vectors in it, and vector type. Creates a new index (folder in this implementation). Add metadata to an index descriptor.json file.

2. _getNamespace_: Return path of a namespace if it exists, else create one.

3. _upsert_: Add a single vector into a namespace (and nearest cluster) given that initial clusters have been created.

4. _upsert_batch_: Argument takes multiple vectors to be upserted. Calls _organise()_ to create clusters if minimum threshold reached.

5. _organise_: Implementation depends on type of index, for IVFFlat, if minimum number of vectors in the namespace exist, create 4 clusters using K-means clustering.

6. _search_: Implementation depends on type of index, for IVFFlat, find nearest cluster centroid for query vector, and do brute-force search within that cluster, return top 3 nearest vectors. 