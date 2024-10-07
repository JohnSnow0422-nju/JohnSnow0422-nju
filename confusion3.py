import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# 读取Excel文档
df = pd.read_excel('2021.xlsx', header=None)

# 将文档的内容存储到一个列表中
documents = df[0].tolist()

# 结巴分词
texts = [' '.join(jieba.cut(doc)) for doc in documents]

# TF-IDF向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# 创建词典和语料库
corpus_texts = [text.split() for text in texts]
dictionary = Dictionary(corpus_texts)
corpus = [dictionary.doc2bow(text) for text in corpus_texts]

# 固定的超参数
alpha = 0.1
beta = 0.01

# 初始化结果字典
results = {
    'num_topics': [],
    'perplexity': []
}

# 计算并记录不同主题数的困惑度
for num_topics in range(1, 51):
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                   alpha=alpha, eta=beta, passes=1000)
    perplexity = lda.log_perplexity(corpus)
    results['num_topics'].append(num_topics)
    results['perplexity'].append(perplexity)

# 将结果保存为DataFrame
results_df = pd.DataFrame(results)

# 保存为Excel文件
results_df.to_excel('Confusion3.xlsx', index=False)

print("Results have been saved to Confusion.xlsx")
