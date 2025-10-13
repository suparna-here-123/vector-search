### VectorSoup - A tiny implementation of vector databases

IVF Flat :
1) First time batch upsert - create clusters
2) Subsequent single-vector upserts - assign to existing clusters
3) PENDING : Every N (TBD) vectors in a given namespace - retrain/reorganise clusters