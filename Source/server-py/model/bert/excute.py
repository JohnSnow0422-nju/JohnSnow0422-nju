from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sentiment import get_sentiment
import pandas as pd
import re
import os
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def generate_charts(topic_num, topic_model, output_dir):
    try:
        topic_similar_heatmap = topic_model.visualize_heatmap(
            n_clusters=topic_num-2)
        topic_similar_heatmap.write_html(f'{output_dir}/主题相似性热力图.html')

        print('生成主题相似性热力图完毕')
    except Exception as err:
        print('生成主题相似性热力图失败', err)
    try:
        term_score_decline = topic_model.visualize_term_rank()
        term_score_decline.write_html(f'{output_dir}/主题术语排名图.html')
        print('生成主题术语排名图完毕')
    except Exception as err:
        print('生成主题层次结构图失败', err)
    try:
        r = topic_model.visualize_hierarchy(top_n_topics=topic_num-1)
        r.write_html(f'{output_dir}/主题层次结构图.html')
        print('生成主题层次结构图完毕')
    except Exception as err:
        print('生成主题层次结构图失败', err)
    try:
        visualize_topics = topic_model.visualize_topics()
        # 可视化结果保存至html中，可以动态显示信息
        visualize_topics.write_html(f'{output_dir}/主题距离图.html')
        print('生成主题距离图完毕')
    except Exception as err:
        print('生成主题距离图失败', err)


def run_model(filter_file, output_dir, seed, topic_num=-1):

    data = []
    umap_model = UMAP(n_neighbors=15, n_components=5,
                      min_dist=0.0, metric='cosine', random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True)
    ctfidf_model = ClassTfidfTransformer()
    for line in pd.read_excel(filter_file)['word_split']:
        line = re.sub(u'\n|\\r', '', line).split(',')
        line = ' '.join(line)
        data.append(line)
    nr_topics = topic_num if topic_num > 0 else 'auto' if topic_num == 0 else None
    topic_model = BERTopic(
        nr_topics=nr_topics,
        language='multilingual',
        embedding_model=embedding_model,    # Step 1 - Extract embeddings
        umap_model=umap_model,              # Step 2 - Reduce dimensionality
        hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
        ctfidf_model=ctfidf_model,
        calculate_probabilities=True
    )

    topic_model.fit_transform(data)
    topic_infos = topic_model.get_topics()
    topic_info_list = []
    for (i, row) in topic_model.get_topic_info().iterrows():
        topic = row['Topic']
        count = row['Count']
        topic_word_list = topic_infos.get(topic)
        topic_value_list = []
        for j in topic_word_list:
            topic_value_list.append(f'{j[0]}:{j[1]}')
        topic_value = ','.join(topic_value_list)
        topic_info_list.append([topic, count, topic_value])
    topic_info_df = pd.DataFrame(topic_info_list, columns=[
        'topic', 'count', 'word_value'])
    with pd.ExcelWriter(f'{output_dir}/主题词分布概率.xlsx') as writer:
        topic_info_df.to_excel(writer, sheet_name='主题词分布概率', index=False)
    documents = []
    for (i, row) in topic_model.get_document_info(data).iterrows():
        belong = row['Topic']
        content = row['Document']
        is_represent = row['Representative_document']
        probability = row['Probability']
        documents.append([belong, content, is_represent, probability])
    document_df = pd.DataFrame(documents, columns=[
        'belong', 'content', 'is_represent', 'probability'])
    with pd.ExcelWriter(f'{output_dir}/不同文档的主题分布.xlsx') as writer:
        document_df.to_excel(writer, sheet_name='不同文档的主题分布', index=False)
    generate_charts(topic_num=len(topic_infos),
                    topic_model=topic_model, output_dir=output_dir)


def get_sentiment_bert(filter_file, doc_topic_file):
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
    res["max_idx"] = res['belong']
    res['word'] = word
    res['lang'] = filter['lang']
    res['sentiment'] = sentiment
    with pd.ExcelWriter(f'{output}/情感分析表.xlsx') as writer:
        res.to_excel(writer, sheet_name='情感分析表', index=False)


def run(id, params, path):
    filter_file = f'{path}/{id}/filter.xlsx'
    seed = params.get('seed', 3407)
    topic_num = params.get('topicNum', -1)
    if (not os.path.exists(f'{os.path.dirname(filter_file)}/主题词分布概率.xlsx') or not os.path.exists(f'{ os.path.dirname(filter_file)}/不同文档的主题分布.xlsx')):
        run_model(seed=seed, filter_file=filter_file,
                  output_dir=os.path.dirname(filter_file), topic_num=topic_num)
    get_sentiment_bert(filter_file=filter_file,
                       doc_topic_file=os.path.dirname(filter_file) + '/不同文档的主题分布.xlsx')
