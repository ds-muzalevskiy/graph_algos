import networkx as nx

#reading data and saving edges
w = int(input())
w_list = []

for i in range(w):
    x = input().split()
    w_list.append(x)

w_list = [int(x[0]) for x in w_list]

e = int(input())
first = []
second = []

for i in range(e):
    x, y = map(int, input().split())
    first.append(x)
    second.append(y)

#creating nx.Graph() and filling it with nodes and edges
G = nx.Graph()

for i in range (len(w_list)):
    G.add_node(i,weight = w_list[i])

for i in range (len(first)):
    G.add_edge(first[i],second[i])


def min_weighted_vertex_cover(G, weight=None):
    """ 
  
    Parameters
    ----------
    G : nx.Graph()

    weight : string, optional (default = None)
        Node weight. If None, then all the nodes have equal weight of -1.
        If string, then it is used as node weight in graph

    Returns
    ----------        
    min_weighted_cover : set
        Returing set of nodes. Summary weight will be not more than 2 times more than summary weight of minimum weightned cover.

    """

    #setting cost, as dictionary with edges with weight attribute
    cost = dict(G.nodes(data=weight, default=1))
    
    #initializing cover as set()
    cover = set()

    #selecting and updating cost until there are uncovered edges exist
    for x, y in G.edges():
        if x in cover or y in cover:
            continue
        if cost[x] <= cost[y]:
            cover.add(x)
            cost[y] -= cost[x]
        else:
            cover.add(y)
            cost[x] -= cost[y]
            
    return cover

#calling min_weighted_vertex_cover() function with "weight" parameter
cover = min_weighted_vertex_cover(G, weight="weight")

#printing result
print(" ".join(str(item) for item in list(cover)))