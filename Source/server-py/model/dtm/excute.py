import os
import pandas as pd
import re
from gensim.corpora import Dictionary
from gensim.models import LdaSeqModel
from model.dtm.coherence import coherence
from sentiment import get_sentiment

filter_filed = ['content', 'time']
filter_sort = ['time']


def get_best_topic_num(coherence_dir):
    coherence_map = {}
    for index, row in pd.read_excel(coherence_dir).iterrows():
        topic_num = row['topic_num']
        coherence_value = row['coherence_value']
        if (topic_num in coherence_map.keys()):
            coherence_map[topic_num] = coherence_map[topic_num] + \
                coherence_value
        else:
            coherence_map[topic_num] = coherence_value
    best_topic_num = 0
    coherence_value = 0
    for topic_num in coherence_map.keys():
        if (float(coherence_map[topic_num]) > coherence_value):
            coherence_value = float(coherence_map[topic_num])
            best_topic_num = int(topic_num)
        else:
            break
    return best_topic_num


def run_model(time_slice, topic_num, filter_file, output_dir, time_sort, seed):
    data = []
    for line in pd.read_excel(filter_file)['word_split']:
        line = re.sub(u'\n|\\r', '', line).split(',')
        data.append(line)
    id2word = Dictionary(data)
    corpus = [id2word.doc2bow(sentence) for sentence in data]
    ldaseq = LdaSeqModel(corpus=corpus, time_slice=time_slice,
                         id2word=id2word, num_topics=topic_num, random_state=seed)  # 将语料库、词典、参数加载入模型中进行训练
    topic_data_print = []
    for time in range(len(time_slice)):
        topic_data = ldaseq.print_topics(time=time)  # 不同时期的主题
        topic_data_print.append(pd.DataFrame(topic_data))
    topic_evolution = []
    for top_index in range(topic_num):
        topic_evolution_data = ldaseq.print_topic_times(
            topic=top_index)  # 不同主题的演变
        topic_evolution.append(pd.DataFrame(topic_evolution_data))
    doc_topic = []
    for index in range(len(data)):
        a = []
        a.extend(ldaseq.doc_topics(index))  # 不同文档的主题概率
        doc_topic.append(a)
    doc_topic_data = pd.DataFrame(doc_topic)
    with pd.ExcelWriter(f'{output_dir}/不同时期主题分布.xlsx') as writer:
        for index in range(len(time_slice)):
            topic_data_print[index].to_excel(
                writer, sheet_name=str(time_sort[index]), index=False)

    with pd.ExcelWriter(f'{output_dir}/不同主题的演变.xlsx') as writer:
        for index in range(len(topic_evolution)):
            topic_evolution[index].to_excel(
                writer, sheet_name='主题'+str(index), index=False)

    with pd.ExcelWriter(f'{output_dir}/不同文档的主题分布.xlsx') as writer:
        doc_topic_data.to_excel(writer, sheet_name='不同文档的主题分布', index=False)


def get_sentiment_dtm(filter_file, doc_topic_file):
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
    res['time'] = filter['time']

    with pd.ExcelWriter(f'{output}/情感分析表.xlsx') as writer:
        res.to_excel(writer, sheet_name='情感分析表', index=False)


def run(id, params, path):
    filter_file = f'{path}/{id}/filter.xlsx'
    time_map = {}
    seed = params.get('seed', 3407)
    topic_num = params.get('topicNum', -1)
    for index, item in pd.read_excel(filter_file).iterrows():
        time = item['time']
        if (time in time_map.keys()):
            time_map[time] = time_map[time] + 1
        else:
            time_map[time] = 1
    time_sort = sorted(list(time_map.keys()))
    time_slice = [time_map[time] for time in time_sort]
    coherence_dir = f'{os.path.dirname(filter_file)}/coherence.xlsx'
    if (not os.path.exists(coherence_dir)):
        coherence(filter_file=filter_file,
                  time_slice=time_slice, time_sort=time_sort, seed=seed)
    best_topic_num = topic_num if topic_num > 0 else get_best_topic_num(
        coherence_dir=coherence_dir)
    if (not os.path.exists(f'{os.path.dirname(filter_file)}/不同文档的主题分布.xlsx') or not os.path.exists(f'{os.path.dirname(filter_file)}/不同时期主题分布.xlsx') or not os.path.exists(f'{os.path.dirname(filter_file)}/不同主题的演变.xlsx')):
        run_model(time_slice=time_slice, topic_num=best_topic_num,
                  filter_file=filter_file, output_dir=os.path.dirname(filter_file), time_sort=time_sort, seed=seed)
    get_sentiment_dtm(filter_file=filter_file,
                      doc_topic_file=f'{os.path.dirname(filter_file)}/不同文档的主题分布.xlsx')
