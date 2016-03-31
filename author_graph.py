import networkx as nx
import community2 as community
import numpy as np
import pandas as pd
from collections import defaultdict

def make_graph_authors(X_train, y_train, info):

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
    
def make_common_neighbors_authors_and_jaccard(sources, targets, G, info): 
    '''
    Return common_neighbors and jaccard similarity
    '''
    
    G2 = G.to_undirected()
    common_neighbors = np.zeros(len(sources))
    jaccard = np.ones(len(sources))
    
    
    all_authors = np.unique([item for sublist in info['authors'] for item in sublist])
    authors_neighbors = {author:G2.neighbors(author) for author in all_authors}
    
    for i in range(len(sources)):
        authors_source = sources[i]
        authors_target = targets[i]

        list_neighbors = [authors_neighbors[author] for author in authors_source]
        neighbors_from = set().union(*list_neighbors)
   
        list_neighbors = [authors_neighbors[author] for author in authors_target]
        neighbors_to = set().union(*list_neighbors)

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

def same_cluster(sources,targets,G):
    partition = community.best_partition(G.to_undirected())
    same_cluster = []
    same_cluster_jaccard = []
    for i,source_authors in enumerate(sources):
        count =0.
        for author1 in sources[i]:
            for author2 in targets[i]:
                if (partition[author1]== partition[author2]):
                    count+=1.
        count/= (len(sources[i])*len(targets[i]))
        same_cluster.append(count)
        A1 = set(sources[i])
        A2 = set(targets[i])
        same_cluster_jaccard.append(float(len(A1.intersection(A2)))/len(A1.union(A2)))
    return same_cluster,same_cluster_jaccard
    
def create_topologic_features_authors(X, G, info, betweeness = True, common_neigh_and_jacc = True, inlinks = True,clusters = True):
    X_ = X.copy()
    X = X.values.astype(int)
    info_authors = info['authors'].to_dict()
    authors_source = [info_authors[source] for source in X[:,0]]
    authors_target = [info_authors[target] for target in X[:,1]]
    if clusters:
        print "creating cluster features"
        scluster,jaccard_cluster = same_cluster(authors_source,authors_target,G)
        X_['Authors Normalized number same cluster'] = scluster
        X_['Authors clusters jaccard'] = jaccard_cluster
        print "cluster features created"
    if betweeness:
        X_['Authors betweeness'] = compute_betweeness_authors(authors_source, authors_target, G)
    
    if common_neigh_and_jacc:
        common_neighbors, jaccard = make_common_neighbors_authors_and_jaccard(authors_source, authors_target, G, info)
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
