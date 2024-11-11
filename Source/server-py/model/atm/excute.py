from model.atm.coherence import coherence
from gensim.models import AuthorTopicModel
from gensim.corpora import Dictionary
import os
import re
import pandas as pd


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
    author_list = []
    data = []
    for (index, row) in pd.read_excel(filter_file).iterrows():
        author_list.append(row['author'])
        data.append([word for word in row['word_split'].split(',')])

    author2doc = {}
    for idx, au in enumerate(author_list):
        author_list_enu = [re.sub(u'\n|\\r', '', i) .strip()
                           for i in au.split(';')]
        for author in author_list_enu:
            if (author in author2doc.keys()):
                author2doc[author].append(idx)
            else:
                author2doc[author] = [idx]
    id2word = Dictionary(data)
    corpus = [id2word.doc2bow(text) for text in data]
    atm = AuthorTopicModel(corpus=corpus, id2word=id2word, author2doc=author2doc,
                           random_state=seed, num_topics=topic_num, passes=10)
    topics = []
    for topic in atm.print_topics(num_topics=topic_num, num_words=20):
        topics.append([i.strip().replace('"', "'")
                       for i in topic[1].split('+')])
    topics_print = pd.DataFrame(topics)
    with pd.ExcelWriter(f'{output_dir}/主题词分布概率.xlsx') as writer:
        topics_print.to_excel(writer, sheet_name='主题词分布概率', index=False)

    author_topics = []
    for author in set(author2doc):
        author_topic_info = atm.get_author_topics(author)
        author_topic_list = []
        for i in author_topic_info:
            author_topic_list.append(f'{i[0]}:{i[1]}')
        author_topics.append([author, ';'.join(author_topic_list)])
    author_topics_print = pd.DataFrame(author_topics)
    with pd.ExcelWriter(f'{output_dir}/不同作者的主题分布.xlsx') as writer:
        author_topics_print.to_excel(
            writer, sheet_name='不同作者的主题分布', index=False)


def run(id, params, path):
    filter_file = f'{path}/{id}/filter.xlsx'
    coherence_dir = f'{os.path.dirname(filter_file)}/coherence.xlsx'
    seed = params.get('seed', 3407)
    topic_num = params.get('topicNum', -1)
    if (not os.path.exists(coherence_dir)):
        coherence(filter_file=filter_file, seed=seed)
    best_topic_num = topic_num if topic_num > 0 else get_best_topic_num(
        coherence_dir=coherence_dir)
    if (not os.path.exists(f'{os.path.dirname(filter_file)}/主题词分布概率.xlsx') or not os.path.exists(f'{os.path.dirname(filter_file)}/不同作者的主题分布.xlsx')):
        run_model(topic_num=best_topic_num, seed=seed, filter_file=filter_file,
                  output_dir=os.path.dirname(filter_file))
