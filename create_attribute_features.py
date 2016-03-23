import networkx as nx
import numpy as np
import pandas as pd
def numb_same_authors(authors1, authors2):
    total = 0
    for author in authors1:
        if author in authors2:
            total += 1
    return total

def create_attribute_features(X,info):
    length = X.shape[0]
    difference_publication_year = np.zeros(length)
    number_same_authors = np.zeros(length)
    self_citation = [0 for x in range(length)]
    same_journal = [0 for x in range(length)]
    
    ### NOT TO OVERFIT WHEN COUNTING : THE INFORMATION ARE AVAILABLE ONLY FOR X_TRAIN ###
#    number_times = X_train.groupby(1).count()
    
    i=-1
    for idx, row in X.iterrows():
        i += 1
        ID1 = int(row[0])
        ID2 = int(row[1])
        
        source = info.loc[ID1]
        target = info.loc[ID2]
        
        ### Difference in publication year
        difference_publication_year[i] = source['year'] - target['year']
        
        ### Number of same authors
        common_authors = numb_same_authors(source['authors'], target['authors'])
        number_same_authors[i] = common_authors
        
        ### Self citation ###
        if common_authors >= 1:
            self_citation[i] = 1
            
        ### Same journal ###
        if source['journal'] == target['journal']:
            same_journal[i] = 1
            
        
    X_ = X.copy()
    X_['Diff publication'] = difference_publication_year
    X_['Number same authors'] = number_same_authors
    X_['Self citation'] = self_citation
    X_['Same journal'] = same_journal
    return X_
