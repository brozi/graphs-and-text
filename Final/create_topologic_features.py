import networkx as nx
import numpy as np
import community
import pandas as pd


def make_graph(X_train, y_train, X_test):
    X_train = pd.concat([X_train, y_train], axis = 1)
    X_train = X_train.values
    G = nx.DiGraph()
    for i in range(X_train.shape[0]):
        source = X_train[i,0]
        target = X_train[i,1]
        G.add_node(source)
        G.add_node(target)
        if X_train[i,-1] == 1:
            G.add_edge(source,target)
            
    X_test = X_test.values
    for i in range(X_test.shape[0]):
        source = X_test[i,0]
        target = X_test[i,1]
        G.add_node(source)
        G.add_node(target)
        
    return G  

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
    X_['Jaccard coefficient'] = make_jaccard(X, G)
    diff_deg, to_deg = compute_diff_inlinks(X, G)
    X_['Difference in inlinks coefficient'] = diff_deg
    X_["Number of times to cited"] = to_deg
    X_['Same cluster'] = same_community(X,G)
    return X_

