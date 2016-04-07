import numpy as np 
import re

def universities_to_keep(authors, universities):
    while('(' in authors and ')' in authors):
        universities.append( authors[authors.find('(')+1 : authors.find(')')] )
        authors = authors[: authors.find('(')] + authors[ authors.find(')')+1 : ]
            
    if '(' in authors:
        universities.append( authors[authors.find('(')+1 : ])
        authors = authors[: authors.find('(')]
    
    return authors, universities


def name_to_keep(author):
    if len(author.split(' ')) <= 1:
        return author
    
    while( author[0] == ' ' and len(author) > 0):
        author = author[1:]
    while( author[-1] == ' ' and len(author) > 0):
        author = author[:-1]
    
    author = author.replace('.', '. ')
    author = author.replace('.  ', '. ')
    name_to_keep = author.split(' ')[0][0] + '. ' + author.split(' ')[-1]

    return name_to_keep

def authors_and_universities(info):
        # Transform concatenated names of authors to a list of authors
    list_authors = []
    list_universities = []

    info['authors'] = info['authors'].replace(np.nan, 'missing')
    for authors in info['authors']:
        if authors != 'missing':
            ### split the different authors
            authors = authors.lower()
            
            ### Find the universities included in the name
            universities = []
            authors, universities = universities_to_keep(authors, universities)
            
            ### Split the authors
            authors = re.split(',|&', authors)
            
            ### For each author, check if university, and store it. Also, keep just the names (To be improved)
            authors_in_article = []      
            for author in authors:
                if author != ' ':
                    authors_in_article.append(name_to_keep(author))
                
            list_universities.append(universities)
            list_authors.append(authors_in_article)
        else:
            list_universities.append(['missing'])
            list_authors.append(['missing'])   

    return list_authors, list_universities