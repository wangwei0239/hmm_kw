# encoding=utf-8
from gensim import corpora
texts = [['human', 'interface', 'computer'],
 ['survey', 'user', 'computer', 'system', 'response', 'time'],
 ['eps', 'user', 'interface', 'system'],
 ['system', 'human', 'system', 'eps'],
 ['user', 'response', 'time'],
 ['trees'],
 ['graph', 'trees'],
 ['graph', 'minors', 'trees'],
 ['graph', 'minors', 'survey']]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus[0])

from gensim import models
tfidf = models.TfidfModel(corpus)
doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow]) # [(0, 0.70710678), (1, 0.70710678)]