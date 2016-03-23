import networkx as nx
import numpy as np
from collections import defaultdict

def make_graph_authors(X_train, y_train, X_test):

    X_train = pd.concat([X_train.ix[:,:2], y_train], axis = 1)
    X_train = X_train.values.astype(int)
    G = nx.DiGraph()
    dict_citation = defaultdict(int)
    
    ### ADD THE AUTHORS ###
    unique_authors = np.unique([item for sublist in info['authors'] for item in sublist])
    for author in unique_authors:
        G.add_node(author)
    
    for i in range(X_train.shape[0]):
        
        if X_train[i,-1] == 1:
            source = X_train[i,0]
            target = X_train[i,1]
            authors_source = info.loc[source]['authors']
            authors_target = info.loc[target]['authors']
            
            for author_source in authors_source:
                for author_target in authors_target:
                    dict_citation[author_source, author_target] += 1
            
    for k,v in dict_citation.iteritems():
        G.add_edge(k[0], k[1], weight = v)    
        
    return G


def compute_betweeness_authors(sources, targets, graph):    
    centrality = nx.degree_centrality(graph)
    betweeness = np.zeros(len(sources))
    
    for i in range(len(sources)):
        authors_source = sources[i]
        authors_target = targets[i]
        
        max_from = max([centrality[x] for x in authors_source]) 
        max_to = max([centrality[x] for x in authors_target])
        
        betweeness[i] = max_from - max_to
        
    return betweeness
    
def make_common_neighbors_authors_and_jaccard(sources, targets, G): 
    '''
    Return common_neighbors and jaccard similarity
    '''
    
    G2 = G.to_undirected()
    common_neighbors = np.zeros(len(sources))
    jaccard = np.ones(len(sources))
    
    for i in range(len(sources)):
        authors_source = sources[i]
        authors_target = targets[i]
        
        neighbors_from = set()
        for author in authors_source:
            neighbors_from |= set(G2.neighbors(author))
            
        neighbors_to = set()
        for author in authors_target:
            neighbors_to |= set(G2.neighbors(author))
        
        inter = len(neighbors_from.intersection(neighbors_to))
        union = len(neighbors_from.union(neighbors_to))
        
        common_neighbors[i] = inter
        if union != 0:
            jaccard[i] = float(inter)/union
        
    return common_neighbors, jaccard
    
def inlinks_authors(sources, targets, G):
    in_degrees = G.in_degree()
    
    max_from = np.array([max([in_degrees[author] for author in authors]) for authors in sources])
    max_to = np.array([max([in_degrees[author] for author in authors]) for authors in targets])
    
    sum_from = np.array([sum([in_degrees[author] for author in authors]) for authors in sources])
    sum_to = np.array([sum([in_degrees[author] for author in authors]) for authors in targets])
    
    median_to = np.array([np.median([in_degrees[author] for author in authors]) for authors in targets])
    
    diff_max = max_to - max_from
    diff_sum = sum_to - sum_from
    
    return diff_max, diff_sum, sum_to, max_to, median_to
    
def create_topologic_features_authors(X, G, info, betweeness = True, common_neigh_and_jacc = True, inlinks = True):
    X_ = X.copy()
    X = X.values.astype(int)
    authors_source = [info.loc[source]['authors'] for source in X[:,0]]
    authors_target = [info.loc[target]['authors'] for target in X[:,1]]
    
    if betweeness:
        X_['Authors betweeness'] = compute_betweeness_authors(authors_source, authors_target, G)
    if common_neigh_and_jacc:
        common_neighbors, jaccard = make_common_neighbors_authors_and_jaccard(authors_source, authors_target, G)
        X_['Authors common neighbors'] = common_neighbors
        X_['Authors jaccard'] = jaccard
    if inlinks:
        diff_max, diff_sum, sum_to, max_to, median_to = inlinks_authors(authors_source, authors_target, G)
        X_['Authors max difference in inlinks'] = diff_max
        X_['Authors sum difference in inlinks'] = diff_sum
        X_['Authors max of times to cited'] = sum_to
        X_['Authors sum of times to cited'] = max_to
        X_['Authors  of times to cited'] = median_to
    
    return X_