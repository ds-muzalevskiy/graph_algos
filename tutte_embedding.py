import fileinput
import numpy as np
from numpy import linalg as linal
import networkx as nx
import numpy.linalg
import random

#reading data from stdin
e = int(input())
lst = []

#writing edges to lst
for i in range(e):
    x, y = map(int, input().split())
    lst.append([x,y])

#identifiyng size of adjacency matrix and creating it, by reading edges
size = len(set([n for e in lst for n in e])) 
adjacency = [[0]*size for i in range(size)]
for sink, source in lst:
    adjacency[sink][source] = 1

#converting adjacency matrix into np.matrix()
mat = np.matrix(adjacency)

#сreating graph using networkx function from_numpy_matrix()
G = nx.from_numpy_matrix(mat)

def spectral(A, dim=2):
    """  
    Parameters
    ----------
    A : np.array()
        Graph matrix
    dim : int
        Dimensionality
     
    Returns
    -------
    r : list
        Sorted list of eigenvectors
    """

    nodes, _ = A.shape

    #Laplacian Matrix
    D = np.identity(nodes, dtype=A.dtype) * np.sum(A, axis=1)
    L = D - A

    eigenvalues, eigenvectors = linal.eig(L)
    
    #sorting and saving least nonzero
    index = np.argsort(eigenvalues)[1 : dim + 1] 
    r = np.real(eigenvectors[:, index])
    return r

def rescale_layout(pos, scale=1):
    """  
    Parameters
    ----------
    pos : dict
        Nodes position dictionary of the cut
    scale : int
        Scaling
     
    Returns
    -------
    pos : dict
    Scaled dictionary of nodes coordinates
    """

    lim = 0  
    #Maximal length for all dimensions
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].mean()
        lim = max(abs(pos[:, i]).max(), lim)

    #changing scale (-scale, scale) for all directions
    if lim > 0:
        for i in range(pos.shape[1]):
            pos[:, i] *= scale / lim
    return pos

def spectral_layout(G, weight="weight", scale=1, dim=2):
    """  
    Parameters
    ----------
    G : nx.Graph()
    weight : string 
       weight of edges
    center : array
        Coordinates pair for cut centring
    dim : int
        Dimension of the cut
     
    Returns
    -------
    pos : dict
        Nodes coordinates dictionary 
    """
    
    #Parameters processing
    empty_graph = nx.Graph()
    empty_graph.add_nodes_from(G)
    G = empty_graph
    
    center = np.zeros(dim)
    
    #Setting the matrix
    A = nx.to_numpy_array(G, weight=weight)
    
    #Calculating the cut
    pos = spectral(A, dim)
    
    #Scaling result
    pos = rescale_layout(pos, scale=scale) + center
    pos = dict(zip(G, pos))
    return pos
  

def tutte_emb(G, outer):
  """
    Parameters
    ----------
    G : nx.Graph() 
        Graph networkx
    outer: list 
        List of nodes from outer plane

    Returns
    -------
    position : dict
    Nodes position of Tutte's Embedding
  """
  
  #dict with nodes positions
  position = {} 
  temporary_graph = nx.Graph()

  for e in outer:
    e1,e2 = e
    temporary_graph.add_edge(e1,e2)
   
  #setting init positions using laplacian
  position = spectral_layout(temporary_graph) 
  
  #identifying nodes of outer plane and remains
  outer_v = temporary_graph.nodes()
  remain_v = [x for x in G.nodes() if x not in outer_v]
  
  #setting linear equations system
  size = len(remain_v)
  q = [[i for i in range(size)] for i in range(size)]
  e = q

  r = [i for i in range(size)]
  w = r
  
  #iterative calculation of nodes positions and their movement to weightned average positions of neighbors 
  for x in remain_v:
    nei = G.neighbors(x) 
    n = len(list(nei))
    a = remain_v.index(x)
  
    q[a][a] = 1
    e[a][a] = 1
    
    for b in nei:
        if b in outer_v:
            w[a] = w[a] + position[b][0]/n
            r[a] = r[a] + position[b][1]/n
    
        else:
            q[a][b] = -(1/float(n))
            e[a][b] = -(1/float(n))
            
  #solving linear equation system getting x and y      
  x = linal.solve(q, w)
  y = linal.solve(e, r)
  
  #getting final positions
  for i in remain_v:
    a = remain_v.index(i)
    position[i] = [x[a],y[a]]
    
  return position

#getting x and y, as keys() and values()
outer = [(0,1), (2,2), (1,2)]
pos = tutte_emb(G, outer)

x = list(pos.keys())
y = numpy.hstack(pos.values())
y = list(np.round(y))

#formatting and result printing
for i, j, z in zip(range(len(x)), range(0,len(y),2), range(1,len(y),2)):
    print(str(x[i]) + ' ' + str(y[j]) + ' ' + str(y[z]))