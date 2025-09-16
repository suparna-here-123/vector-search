import numpy as np

custom = [('_id', 'i4'), ('vector', int, (4,)), ('metadata', dict)]

a1 = np.array([1,2,3,4])
a2 = np.array([1,2,5,4])
a3 = np.array([1,12,3,4])

v1 = np.array([(1, a1, {'docId' : 15, 'type' : 'political'})], dtype=custom)
v2 = np.array([(2, a2, {'docId' : 13, 'type' : 'history'})], dtype=custom)
v3 = np.array([(3, a3, {'docId' : 15, 'type' : 'economics'})], dtype=custom)


# np.save(f'Indexes/IVFFlat_ABC/product/cluster_1/vector_1.npy', v1)
# np.save(f'Indexes/IVFFlat_ABC/product/cluster_1/vector_2.npy', v2)
# np.save(f'Indexes/IVFFlat_ABC/product/cluster_1/vector_3.npy', v3)

for i in range(1,4) :
    a, b, c = (np.load(f'Indexes/IVFFlat_ABC/product/cluster_1/vector_{i}.npy', allow_pickle=True))[0]
    print(a)