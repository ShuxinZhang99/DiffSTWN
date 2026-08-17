# -*- coding: utf-8 -*-

# import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from math import sqrt
# from scipy.sparse import csgraph
# adj = csgraph.laplacian(adj, normed=True)



class GetLaplacian:
	def __init__(self,adjacency):
		self.adjacency=adjacency

	#Method1 to calculate laplacian
	def get_normalized_adj(self, station_num):
		I = np.matrix(np.eye(station_num))
		A_hat = self.adjacency + I
		D_hat = np.array(np.sum(A_hat, axis=0))[0]
		D_hat_sqrt = [sqrt(x) for x in D_hat]
		D_hat_sqrt = np.array(np.diag(D_hat_sqrt))
		D_hat_sqrtm_inv = np.linalg.inv(D_hat_sqrt)  # 开方后求逆即为矩阵的-1/2次方
		# D_A_final=D_hat**-1/2 * A_hat *D_hat**-1/2
		D_A_final = np.dot(D_hat_sqrtm_inv, A_hat)
		D_A_final = np.dot(D_A_final, D_hat_sqrtm_inv)
		# print(D_A_final.shape)
		return np.array(D_A_final,dtype="float32")

	# Method2 to calculate laplacian
	def normalized_adj(self):
		adj = sp.coo_matrix(self.adjacency)
		rowsum = np.array(adj.sum(1))
		d_inv_sqrt = np.power(rowsum, -0.5).flatten()
		d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
		d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
		normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
		normalized_adj = normalized_adj.astype(np.float32)
		return normalized_adj


	def sparse_to_tuple(self,mx):
		mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		L = tf.SparseTensor(coords, mx.data, mx.shape)
		return tf.sparse.reorder(L)


	def calculate_laplacian(self):
		adj = self.normalized_adj(np.array(self.adjacency) + sp.eye(np.array(self.adjacency).shape[0]))
		adj = sp.csr_matrix(adj)
		adj = adj.astype(np.float32)
		return self.sparse_to_tuple(adj)