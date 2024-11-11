from model.filter import filter_data
from model.dtm.excute import run as dtm
from model.lda.excute import run as lda
from model.bert.excute import run as bert
from model.atm.excute import run as atm
from flask import Flask, request
import json
import os
import requests

app = Flask(__name__)
waitting_list = []
is_excute = False

fun_map = {
    'dtm': dtm,
    'lda': lda,
    'bert': bert,
    'atm': atm
}
orderList = []


def upload_files(path, id):
    files = {}
    for (filename) in os.listdir(f'{path}/{id}'):
        file_path = f'{path}/{id}/{filename}'
        if (os.path.isfile(file_path) and not filename == 'source.xlsx'):
            files[filename] = (f'{filename}', open(
                file_path, 'rb'), "multipart/form-data")
    i = 0
    while i < 3:
        try:
            requests.put('http://43.142.133.76/model/files',
                         files=files, data={'id': id})
            break
        except Exception as err:
            i += 1
            print('正在重连...') if i < 2 else print(err)


def process(target):
    # print(target)
    id = str(target.get('id'))
    params = target.get('params', {})
    config = target.get('config', {})
    path = target.get('path')
    filter_data(id, config, path)
    model = params.get('model', 'lda')
    if __name__ == '__main__':
        fun_map[model](id=id, params=params, path=path)
        upload_files(path=path, id=id)


def run():
    try:
        global is_excute
        global waitting_list
        print('run', is_excute, waitting_list, '\n\n')
        if (len(waitting_list) == 0 or is_excute == True):
            return
        is_excute = True
        target = waitting_list.pop(0)
        process(target)
        is_excute = False
        run()
    except Exception as err:
        print(err)
        is_excute = False
        run()


@app.route('/process', methods=['post'])
def add():
    try:
        params = request.get_json()['params']
        config = request.get_json()['config']
        path = request.get_json()['path']
        id = request.get_json()['_id']
        global waitting_list
        waitting_list.append(
            {"params": params, "config": config, "path": path, "id": id})
        run()
        return json.dumps({"code": 'success', "len": len(waitting_list)})
    except Exception as err:
        print(err)


if __name__ == '__main__':
    # job_threading = threading.Thread(target=init)
    # job_threading.start()
    app.run(host='0.0.0.0', port=3002, debug=False)
