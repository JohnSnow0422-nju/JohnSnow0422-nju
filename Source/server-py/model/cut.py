import hanlp
import re
import os

import hanlp.pretrained

en = hanlp.load(hanlp.pretrained.tok.UD_TOK_MMINILMV2L12)
zh = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
stopword_list = []
replace_dic = {}


def init_dictionary(path):
    print('词表目录', f'{path}/dict')
    try:
        combine = open(f'{path}/dict/combine.txt', encoding='utf-8').read() if os.path.exists(
            f'{path}/dict/combine.txt') else ''
        force = open(f'{path}/dict/force.txt', encoding='utf-8').read() if os.path.exists(
            f'{path}/dict/force.txt') else ''
        tok_dic = set()
        force_dic = set()
        en.dict_force = force_dic
        en.dict_combine = tok_dic
        zh.dict_force = force_dic
        zh.dict_combine = tok_dic
        try:
            for line in combine.splitlines():
                line = re.sub(u'\n|\\r', '', line).split(' ')
                tok_dic.add(line[0])
            for line in force.splitlines():
                line = re.sub(u'\n|\\r', '', line).split(' ')
                force_dic.add(line[0])
        except:
            return
        en.dict_force = force_dic
        en.dict_combine = tok_dic
        zh.dict_force = force_dic
        zh.dict_combine = tok_dic
    except Exception as err:
        print('初始化词典失败', err)

    global stopword_list
    stopword_list = []
    try:
        stop = open(f'{path}/dict/stopword.txt', encoding='utf-8').read() if os.path.exists(
            f'{path}/dict/stopword.txt') else ''
        for line in stop.splitlines():
            line = re.sub(u'\n|\\r', '', line).replace(' ', '')
            stopword_list.append(line)
    except:
        print('初始化停用词失败')
    global replace_dic
    replace_dic = {}
    try:
        replace = open(f'{path}/dict/replace.txt', encoding='utf-8').read(
        ) if os.path.exists(f'{path}/dict/replace.txt') else ''
        for line in replace.splitlines():
            line = re.sub(u'\n|\\r', '', line).split(' ')
            replace_dic[line[0]] = line[1]
    except:
        print('初始化替换词失败')


def word_cut(word, lang='en'):
    rs = []
    tok = zh if lang == 'zh' else en
    try:
        for seg_word in tok(word):
            word_item = re.sub(r'http[s]?://\S+', '', seg_word)
            word_item = re.compile(
                "[^\\u4e00-\\u9fa5^a-zA-Z^0-9]").sub("", word_item)
            word_item = re.sub(r'\s+', ' ', word_item)
            if (word_item not in stopword_list):
                if (word_item in replace_dic.keys()):
                    word_item = replace_dic[word_item]
                rs.append(word_item)
        rs = [i for i in rs if i]
        print((' ').join(rs))
        return rs
    except:
        return rs
