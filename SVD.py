import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from math import sqrt

#Load ml-100k into pandas with labels.
df = pd.read_csv('data/ml-100k.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])


#Declare number of users and movies.
n_users = df.user_id.unique().shape[0]  #943
n_movies = df.movie_id.unique().shape[0]  #1682

#Creates a train test split of 75/25.
train_data, test_data = train_test_split(df, test_size=0.25)

#Populates a train and test matrix (user_id x movie_id), containing ratings.
train_data_matrix = np.zeros((n_users, n_movies))
for line in train_data.itertuples():
    #[user_id index, movie_id index] = given rating.
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((n_users, n_movies))
for line in test_data.itertuples():
    #[user_id index, movie_id index] = given rating.
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


sparsity=round(1.0-len(df)/float(n_users*n_movies),3)
print 'Sparsity level of MovieLens 100K Dataset is ', sparsity

#Root Mean Squared Error for validation.
def rmse(pred, test):
    pred = pred[test.nonzero()].flatten() 
    test = test[test.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, test))

#Get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print 'User-based CF MSE: ', rmse(X_pred, test_data_matrix)