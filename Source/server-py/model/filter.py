from model.cut import word_cut, init_dictionary
from opencc import OpenCC
import pandas as pd
import os

converter = OpenCC('t2s')


def has_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)


def filter_data(dirId, config, path):
    try:
        pathDir = '/'.join([path, dirId])
        if (os.path.exists(pathDir + '/' + 'filter.xlsx')):
            return
        sourceFile = pathDir + '/' + 'source.xlsx'
        review = []
        sort = config.get('sort', '')
        field_list = config.get('field', [])
        fileData = pd.read_excel(sourceFile)
        init_dictionary(path=path)
        for index, row in fileData.iterrows():
            lang = 'zh' if has_chinese(row['content']) else 'en'
            rs = word_cut(converter.convert(row['content']), lang=lang)
            if (len(rs) > 0):
                rs_row = []
                for key in field_list:
                    rs_row.append(row[key])
                rs_row.append((',').join(rs))
                rs_row.append(lang)
                review.append(rs_row)
        if (bool(sort)):
            index = field_list.index(sort)
            review = sorted(review, key=lambda x: x[index])
        filter_data = pd.DataFrame(
            review, columns=field_list + ['word_split', 'lang'])
        print('filter.xlsx输出地址', f'{pathDir}/filter.xlsx')
        with pd.ExcelWriter(f'{pathDir}/filter.xlsx') as writer:
            filter_data.to_excel(writer, sheet_name='sheet', index=False)
    except Exception as err:
        print('filter-error', err)
