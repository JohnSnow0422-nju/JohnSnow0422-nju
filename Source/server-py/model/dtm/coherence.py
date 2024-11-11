from gensim.models import LdaSeqModel, CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
import re
import os

# 一致性确定最优主题数


def coherence(filter_file, time_slice, time_sort, seed):
    data = []
    for line in pd.read_excel(filter_file)['word_split']:
        line = re.sub(u'\n|\\r', '', line).split(',')
        data.append(line)
    id2word = Dictionary(data)
    corpus = [id2word.doc2bow(sentence) for sentence in data]
    rs_list = []
    for t_num in range(4, 11):
        print('first', t_num)
        # DTM主题建模，训练阶段,em_max_iter设置为5，是为了降低迭代次数，节约时间
        ldaseq = LdaSeqModel(
            corpus=corpus, id2word=id2word, time_slice=time_slice, num_topics=t_num, random_state=seed)
    #     # 分阶段计算，如果阶段大于3，则直接替换3为具体数量
        for t_time in range(0, len(time_slice)):
            print('second', t_time)
            topics_dtm = ldaseq.dtm_coherence(time=t_time)   # 不同阶段的主题识别效果
            cm_DTM = CoherenceModel(
                topics=topics_dtm, texts=data, dictionary=id2word, coherence='c_v', processes=1)
            rs_list.append([str(t_num), str(time_sort[t_time]),
                           str(cm_DTM.get_coherence())])

    coh = pd.DataFrame(rs_list, columns=[
                       'topic_num', 'time', 'coherence_value'])
    with pd.ExcelWriter(f'{os.path.dirname(filter_file)}/coherence.xlsx') as writer:
        coh.to_excel(writer, sheet_name='sheet', index=False)
