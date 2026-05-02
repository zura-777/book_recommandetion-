import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class SVDModel:
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_map = {}
        self.item_map = {}
        self.u_vecs = None
        self.i_vecs = None
        self.global_mean = 0

    def fit(self, df):
        self.user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
        self.item_map = {iid: i for i, iid in enumerate(df['book_id'].unique())}
        u_indices = df['user_id'].map(self.user_map).values
        i_indices = df['book_id'].map(self.item_map).values
        ratings = df['rating'].values
        self.global_mean = ratings.mean()
        matrix = csr_matrix((ratings - self.global_mean, (u_indices, i_indices)), 
                           shape=(len(self.user_map), len(self.item_map)))
        k = min(self.n_factors, min(matrix.shape) - 1)
        u, s, vt = svds(matrix, k=k)
        self.u_vecs = u @ np.diag(np.sqrt(s))
        self.i_vecs = np.diag(np.sqrt(s)) @ vt

    def predict(self, uid, iid):
        u_idx = self.user_map.get(uid)
        i_idx = self.item_map.get(iid)
        if u_idx is None or i_idx is None:
            return self.global_mean
        prediction = self.global_mean + np.dot(self.u_vecs[u_idx], self.i_vecs[:, i_idx])
        return float(np.clip(prediction, 1, 5))
