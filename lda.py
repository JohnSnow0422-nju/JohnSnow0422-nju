import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# 加载停用词表
def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

# 加载和预处理数据
def load_data(file_path):
    df = pd.read_excel(file_path, header=None, names=['text'])
    return df['text'].tolist()

def preprocess_text(text, stopwords):
    words = list(jieba.cut(text))
    return [word for word in words if word not in stopwords and len(word.strip()) > 0]

def preprocess_corpus(corpus, stopwords):
    return [preprocess_text(text, stopwords) for text in corpus]

# 使用 TF-IDF 向量化
def vectorize_corpus(corpus):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    return tfidf_matrix, terms

# 准备 gensim 的词典和语料库
def prepare_corpus(tfidf_matrix, terms):
    corpus = gensim.matutils.Sparse2Corpus(tfidf_matrix, documents_columns=False)
    id2word = {i: word for i, word in enumerate(terms)}
    dictionary = corpora.Dictionary.from_corpus(corpus, id2word=id2word)
    return dictionary, corpus

# 计算一致性分数
def compute_coherence_values(dictionary, corpus, start, limit, step, texts):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=dictionary,
                                                num_topics=num_topics,
                                                alpha=0.1,
                                                eta=0.01,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=1000,
                                                per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

# 绘制一致性分数
def plot_coherence_values(start, limit, step, coherence_values):
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()

def save_coherence_scores(coherence_values, start, step, file_path):
    topics = range(start, start + len(coherence_values) * step, step)
    df = pd.DataFrame({'Number of Topics': topics, 'Coherence Score': coherence_values})
    df.to_excel(file_path, index=False)

def main():
    # 文件名
    file_name = '/Users/johnsnow/Desktop/科技政策/2023.xlsx'
    output_file = '/Users/johnsnow/Desktop/科技政策/coherence.xlsx'
    stopwords_file = '/Users/johnsnow/Desktop/科技政策/stopwords.txt'  # 请将路径替换为实际停用词文件路径

    # 加载停用词
    stopwords = load_stopwords(stopwords_file)

    # 加载数据
    data = load_data(file_name)

    # 预处理语料库
    processed_corpus = preprocess_corpus(data, stopwords)

    # 向量化语料库
    tfidf_matrix, terms = vectorize_corpus(processed_corpus)

    # 准备 gensim 的词典和语料库
    dictionary, corpus = prepare_corpus(tfidf_matrix, terms)

    # 计算一致性分数
    start, limit, step = 1, 51, 1
    model_list, coherence_values = compute_coherence_values(dictionary, corpus, start, limit, step, processed_corpus)

    # 绘制一致性分数
    plot_coherence_values(start, limit, step, coherence_values)

    # 输出一致性结果到文件
    save_coherence_scores(coherence_values, start, step, output_file)

    # 打印一致性结果
    for i, coherence in enumerate(coherence_values):
        num_topics = start + i * step
        print(f'Number of Topics: {num_topics}, Coherence Score: {coherence}')

if __name__ == "__main__":
    main()
