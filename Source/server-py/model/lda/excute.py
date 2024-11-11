from model.lda.coherence import coherence
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import os
import pandas as pd
import re
from sentiment import get_sentiment


def get_best_topic_num(coherence_dir):
    best_topic_num = 0
    coherence_value = 0
    for index, row in pd.read_excel(coherence_dir).iterrows():
        if (row['coherence_value'] > coherence_value):
            coherence_value = float(row['coherence_value'])
            best_topic_num = int(row['topic_num'])
        else:
            break
    return best_topic_num


def run_model(topic_num, filter_file, output_dir, seed):
    data = []
    for line in pd.read_excel(filter_file)['word_split']:
        line = re.sub(u'\n|\\r', '', line).split(',')
        data.append(line)
    id2word = Dictionary(data)
    corpus = [id2word.doc2bow(sentence) for sentence in data]
    lda = LdaModel(
        corpus=corpus, id2word=id2word, num_topics=topic_num, passes=10, random_state=seed)
    topic_data = []
    for topic in lda.print_topics(num_topics=topic_num, num_words=20):
        topic_data.append([i.strip().replace('"', "'")
                          for i in topic[1].split('+')])
    topic_data_print = pd.DataFrame(topic_data)
    doc_topic_data = lda.get_document_topics(bow=corpus)
    doc_topic = []
    for row in doc_topic_data:
        arr = [0] * topic_num
        for item in row:
            arr[item[0]] = item[1]
        doc_topic.append(arr)
    doc_topic_print = pd.DataFrame(doc_topic)
    with pd.ExcelWriter(f'{output_dir}/主题词分布概率.xlsx') as writer:
        topic_data_print.to_excel(writer, sheet_name='主题词分布概率', index=False)
    with pd.ExcelWriter(f'{output_dir}/不同文档的主题分布.xlsx') as writer:
        doc_topic_print.to_excel(writer, sheet_name='不同文档的主题分布', index=False)


def get_sentiment_lda(filter_file, doc_topic_file):
    filter = pd.read_excel(filter_file)
    topic = pd.read_excel(doc_topic_file)
    output = os.path.dirname(filter_file)
    sentiment = []
    word = []
    for index, row in filter.iterrows():
        word_key = 'word_split' if row['lang'] == 'zh' else 'content'
        sentiment.append(get_sentiment(row[word_key], row['lang']))
        word.append(row[word_key])
    res = pd.DataFrame(topic)
    res["max_idx"] = res.idxmax(axis=1, numeric_only=True)
    res['word'] = word
    res['lang'] = filter['lang']
    res['sentiment'] = sentiment
    with pd.ExcelWriter(f'{output}/情感分析表.xlsx') as writer:
        res.to_excel(writer, sheet_name='情感分析表', index=False)


def run(id, params, path):
    filter_file = f'{path}/{id}/filter.xlsx'
    coherence_dir = f'{os.path.dirname(filter_file)}/coherence.xlsx'
    seed = params.get('seed', 3407)
    topic_num = params.get('topicNum', -1)
    if (not os.path.exists(coherence_dir)):
        coherence(filter_file=filter_file, seed=seed)
    best_topic_num = topic_num if topic_num > 0 else get_best_topic_num(
        coherence_dir=coherence_dir)
    if (not os.path.exists(f'{os.path.dirname(filter_file)}/主题词分布概率.xlsx') or not os.path.exists(f'{os.path.dirname(filter_file)}/不同文档的主题分布.xlsx')):
        run_model(topic_num=best_topic_num, seed=seed, filter_file=filter_file,
                  output_dir=os.path.dirname(filter_file))
    get_sentiment_lda(filter_file=filter_file,
                      doc_topic_file=f'{os.path.dirname(filter_file)}/不同文档的主题分布.xlsx')
