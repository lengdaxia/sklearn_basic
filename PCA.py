from sklearn.decomposition import PCA
import numpy as np


print(' ')
print('*********************  basic user  *********************')
print(' ')

X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])

pca = PCA(n_components=2)
pca.fit(X)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)



print(' ')
print('*********************  face_recogntion use pca *********************')
print(' ')


