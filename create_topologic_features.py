import networkx as nx
import numpy as np
import community2 as community

def same_community(X,G):
    partition = community.best_partition(G.to_undirected())
    return np.array([1 if partition[x[0]]== partition[x[1]] else 0 for x in X])

def compute_betweeness_array(X, graph):    
    centrality = nx.degree_centrality(graph)  
    centr = [centrality[x[0]] - centrality[x[1]] for x in X[:]]
    return np.array(centr)

def make_common_neighbors(X, G): 
    G2 = G.to_undirected()
    common_neighbors = [len(set(G2.neighbors(x[0])).intersection(G2.neighbors(x[1]))) for x in X[:]]
    return np.array(common_neighbors)

def make_jaccard(X, G):   
    total_jaccard = nx.jaccard_coefficient(G.to_undirected(), [(x[0],x[1]) for x in X[:]])
    jaccard = [jac for u, v, jac in total_jaccard]
    return np.array(jaccard)

#Difference in inlinks between papers
def compute_diff_inlinks(X, graph):
    in_degrees=graph.in_degree()
    diff_deg = [in_degrees[x[1]] - in_degrees[x[0]] for x in X[:]]
    to_deg = [in_degrees[x[1]] for x in X[:]]
    return diff_deg, to_deg

def create_topologic_features(X, G):
    X_ = X.copy()
    X = X.values
    
    X_['Betweeness centrality'] = compute_betweeness_array(X, G)
    X_['Number common neighbours'] = make_common_neighbors(X, G)
    X_['Jaccard coefficienf'] = make_jaccard(X, G)
    diff_deg, to_deg = compute_diff_inlinks(X, G)
    X_['Difference in inlinks coefficient'] = diff_deg
    X_["Number of times to cited"] = to_deg
    X_['Same cluster'] = same_community(X,G)
    return X_

