import networkx
import numpy as np

def create_tree(N,p):
    while True:
        G = networkx.erdos_renyi_graph(N,p)
        if networkx.number_connected_components(G)==1:
            return networkx.minimum_spanning_tree(G)

def make_directed(A):
    A = A.copy()
    def rec(i):
        children = np.where(A[i])[0]
        for j in children:
            A[j, i] = 0
            rec(j)
    rec(0)
    return A

def create_random_adjacency_matrix(N, p):
    G = create_tree(N, p)
    center = networkx.algorithms.distance_measures.center(G)[0]
    A = np.asarray(networkx.to_numpy_matrix(G)).astype(int)
    ind = np.array([center] + list(range(center)) + list(range(center+1,N)))
    return make_directed(A[np.ix_(ind,ind)])

def adjmat_to_child_list(A):
    return [list(np.where(a)[0]) for a in A]

def sample_tree(N, p):
    A = create_random_adjacency_matrix(N, p)
    return adjmat_to_child_list(A)