import numpy as np
import networkx as nx
from scipy.sparse import csgraph

#reading data and setting adjacency matrix
e = int(input())
first = []
spectral = []


for i in range(e):
    x, y = map(int, input().split())
    first.append([x,y])

size = len(set([n for e in first for n in e])) 
adjacency = [[0]*size for _ in range(size)]
for sink, source in first:
    adjacency[sink][source] = 1

#convering into np.matrix() and creating graph using nx.from_numpy_matrix()
mat = np.matrix(adjacency)
G = nx.from_numpy_matrix(mat)

m_adjacency = np.array(nx.to_numpy_matrix(G))

#calculating laplacian
M = csgraph.laplacian(m_adjacency)

#finding eigenvalues and eigenvectors, sorting, selecting second least value
(w_, v_) = np.linalg.eig(M)
index = np.argsort(w_)[1]

v_part = v_[:, index]

#sorting nodes using new index
new_sort = sorted(list(G.nodes), key=lambda x: v_part[list(G.nodes).index(x)])

#calculating "phi" metric for cut evaluation
def phi(G, V):
    
    """ 
  
    Parameters
    ----------
    G : nx.Graph()

    V : list
        List of nodes, sorted using new index

    Returns
    ----------
    phi : float
        phi = e*len(G)/((len(G) - len(V))*len(V))

    """

    e = len([e for e in G.edges
             if e[0] in V and e[1] not in V
             or e[1] in V and e[0] not in V])
    phi =  float(e*len(G)/((len(G) - len(V))*len(V)))
    return phi

#iterating over graph, calculating phi by checking different cuts
for i in range(1,len(G)):
    d = phi(G, new_sort[:i])
    if d < min:
        min = d
        spectral_cuts = [new_sort[:i] if i <= len(G)//2 else new_sort[i:]]
    if d == min:
        spectral_cuts.append(new_sort[:i] if i <= len(G)//2 else new_sort[i:])

#printing minumum cut
print(*sorted(spectral_cuts)[0])