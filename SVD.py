import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds


df = pd.read_csv('data/ml-100k.data', sep='\t', names=['movie_id', 'item_id', 'rating', 'timestamp'])


#Declare number of users and movies.
n_users = df.user_id.unique().shape[0]  #943
n_movies = df.movie_id.unique().shape[0]  #1682

sparsity=round(1.0-len(df)/float(n_users*n_movies),3)
print 'The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%'

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print 'User-based CF MSE: ', rmse(X_pred, test_data_matrix)