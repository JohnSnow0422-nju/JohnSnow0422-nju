from gensim.models import CoherenceModel, atmodel
from gensim.corpora import Dictionary
import pandas as pd
import os
import re


def coherence(filter_file, seed):
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
    rs_list = []
    for t_num in range(4, 11):
        print('first', t_num)
        atm_model = atmodel.AuthorTopicModel(corpus=corpus, num_topics=t_num, passes=10,
                                             author2doc=author2doc, id2word=id2word, random_state=seed)
        coherencemodel = CoherenceModel(
            model=atm_model, texts=data, dictionary=id2word, coherence='c_v', processes=1)
        rs_list.append([str(t_num), str(coherencemodel.get_coherence())])
    coh = pd.DataFrame(rs_list, columns=[
        'topic_num', 'coherence_value'])
    with pd.ExcelWriter(f'{os.path.dirname(filter_file)}/coherence.xlsx') as writer:
        coh.to_excel(writer, sheet_name='sheet', index=False)
