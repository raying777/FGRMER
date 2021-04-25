import torch
import numpy as np

def gen_A(num_classes, t, adj_file):
    #import pickle
    #result = pickle.load(open(adj_file, 'rb'))
    _adj = np.load(adj_file)
    #_adj = result['adj']
    #_nums = result['nums']
    #_nums = _nums[:, np.newaxis]
    #_adj = _adj / _nums
#    _adj[_adj < t] = 0
#    _adj[_adj >= t] = 1
#    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
