import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# 读取停用词表
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

stopwords = load_stopwords('/Users/johnsnow/Desktop/科技政策/stopwords.txt')  # 请将路径替换为实际停用词文件路径

# 读取Excel文档
df = pd.read_excel('2021.xlsx', header=None)

# 假设文档的内容在无标题的第一列中
documents = df[0].tolist()

# 结巴分词并去除停用词
texts = []
for doc in documents:
    words = jieba.cut(doc)
    filtered_words = [word for word in words if word not in stopwords]
    texts.append(' '.join(filtered_words))

# TF-IDF向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# 创建词典和语料库
corpus_texts = [text.split() for text in texts]
dictionary = Dictionary(corpus_texts)
corpus = [dictionary.doc2bow(text) for text in corpus_texts]

# 设置LDA模型参数
num_topics = 12
alpha = 1
beta = 0.01
passes = 1000

# 训练LDA模型
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
               alpha=alpha, eta=beta, passes=passes)

# 输出主题-主题词矩阵
topic_terms_matrix = []
for i in range(num_topics):
    terms = lda.get_topic_terms(i, topn=30)
    topic_terms_matrix.append([dictionary[word_id] for word_id, prob in terms])

# 将主题-主题词矩阵转换为DataFrame并输出
topic_terms_df = pd.DataFrame(topic_terms_matrix)
print("主题-主题词矩阵：")
print(topic_terms_df)

# 保存结果到Excel
output_path = '/Users/johnsnow/Desktop/科技政策/LDA_Results.xlsx'
with pd.ExcelWriter(output_path) as writer:
    topic_terms_df.to_excel(writer, sheet_name='Topic-Terms', index=False)
