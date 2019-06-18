
#资料https://blog.csdn.net/Yolen_Chan/article/details/84934928

#if-idf的计算
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
           [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
           [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
           [(0, 1.0), (4, 2.0), (7, 1.0)],
           [(3, 1.0), (5, 1.0), (6, 1.0)],
           [(9, 1.0)],
           [(9, 1.0), (10, 1.0)],
           [(9, 1.0), (10, 1.0), (11, 1.0)],
           [(8, 1.0), (10, 1.0), (11, 1.0)]]

from gensim import models
tfidf=models.TfidfModel(corpus)
print(tfidf)

vec=[(0,1),(4,1)]
print(tfidf[vec])

#%%
#通过对tfidf转换对语料库进行索引，进行相似度查询
from gensim  import similarities

index=similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)
print(index)

sims=index[tfidf[vec]]
print(list(enumerate(sims)))
#%%进行清洗，去除停用词，大小写转换
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]


from collections import defaultdict
stoplist=set('for a of the and to in'.split())

texts=[
    [word for word in document.lower().split() if word not in stoplist] for document in documents
]

#注意计数的方式
frequency=defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts=[[token for token in text if frequency[token]>1] for text in texts]
print(texts)


#%%使用gensim做自然语言处理的一般思路是：使用（处理）字典 ----> 生成（处理）语料库 ----> 自然语言处理（tf-idf的计算等)
#1生成字典,并保存字典
from gensim import corpora
dictionary=corpora.Dictionary(texts)
#dictionary.save('/deerwester.dict')
print(dictionary)

#%%将文本中的词与数字做一一映射
print(dictionary.token2id)
#%%要标记文档实际转换为向量,根据字典里面的词返回稀疏矩阵
new_doc="Human computer response human"
new_vec=dictionary.doc2bow(new_doc.lower().split())
print(new_vec) #第二个维度表示出现的次数

#%%对语料库进行处理--将次列表转换成为稀疏词袋向量
print(texts)
corpus=[]
for text in texts:
    corpus.append(dictionary.doc2bow(text))

print(corpus)
#%%根据稀疏词袋向量变成tf-idf向量
from gensim import models
#通过训练对语料库进行初始化
tfidf_modle=models.TfidfModel(corpus=corpus,dictionary=dictionary)
tfidf_modle.save('test_tfidf.model')
tfidf_modle=models.TfidfModel.load('test_tfidf.model')

print(tfidf_modle)
corpus_tfidf=[tfidf_modle[doc] for  doc in corpus]
print(corpus_tfidf)
print()
#%%对整个语料库进行转换

print(tfidf_modle[corpus])
corpus_tfidf=tfidf_modle[corpus]
#%%Lsi主题模型
lsi=models.LsiModel(corpus_tfidf,num_topics=20,id2word=dictionary)
corpus_lsi=lsi[corpus_tfidf]
nodes=list(corpus_lsi)
#print(nodes)
print(lsi.print_topic(20))
