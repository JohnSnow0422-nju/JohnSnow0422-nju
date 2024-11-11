from gensim.models import ldamodel, CoherenceModel
import pandas as pd
from gensim import corpora
import os
import re


def coherence(filter_file, seed):
    data = []
    for line in pd.read_excel(filter_file)['word_split']:
        line = re.sub(u'\n|\\r', '', line).split(',')
        data.append(line)
    id2word = corpora.Dictionary(data)
    corpus = [id2word.doc2bow(sentence) for sentence in data]
    rs_list = []
    for t_num in range(4, 11):
        print('first', t_num)
        lda_model = ldamodel.LdaModel(corpus=corpus, num_topics=t_num,
                                      id2word=id2word,  passes=10, random_state=seed)
        coherencemodel = CoherenceModel(
            model=lda_model, texts=data, dictionary=id2word, coherence='c_v', processes=1)
        rs_list.append([str(t_num), str(coherencemodel.get_coherence())])
    coh = pd.DataFrame(rs_list, columns=[
                       'topic_num', 'coherence_value'])
    with pd.ExcelWriter(f'{os.path.dirname(filter_file)}/coherence.xlsx') as writer:
        coh.to_excel(writer, sheet_name='sheet', index=False)
