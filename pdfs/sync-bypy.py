from bypy import ByPy
import re
from tqdm import tqdm
import os
from joblib import Parallel, delayed
import urllib3


def preprocess(f, root):
    """预处理
     - 字母小写
     - 删除两侧空格
     - 出现同名则删除该论文
    """
    bs, ext = os.path.splitext(f)
    bs = bs.lower().strip()
    af = f'{bs}{ext}'
    af = re.sub(r'[/:*<>?|]', '-', af)
    af = re.sub(r'[\\]', '', af)
    if f != af:
        if os.path.exists(os.path.join(root, af)):
            os.remove(os.path.join(root, f))
        else:
            os.rename(os.path.join(root, f), os.path.join(root, af))
    return af


# 只在当前目录下同步 pdf 文件
cur_dir = os.path.dirname(os.path.abspath(__file__))
sync_f = os.path.join(cur_dir, '.sync')


def generate_fs_list():
    if os.path.exists(sync_f):
        with open(sync_f, encoding='utf-8') as r:
            sync_fs = set([i.strip() for i in r.readlines()])
            # 忽略百度云删除的情况，全量同步删除 .sync 即可
    else:
        sync_fs = {}
    for root, _, fs in os.walk(cur_dir):
        for f in fs:
            if not f.endswith('.pdf'):
                continue
            f = preprocess(f, root)
            if f in sync_fs:
                continue
            rel_dir = root.replace(cur_dir, '').lstrip('/')
            yield f, os.path.join(root, f), os.path.join(rel_dir, f)


# 同步到百度云
bp = ByPy()


def upload(f, absf, tgtf):
    urllib3.disable_warnings()
    src = os.path.join(absf)
    tgt = os.path.join('paper-pdfs', tgtf)
    bp.upload(src, tgt)
    return f


def group(iters, number):
    res = []
    for item in iters:
        res.append(item)
        if len(res) > number:
            yield res
            res = []
    yield res


# print(list(generate_fs_list()))
with open(sync_f, 'a', encoding='utf-8') as w:
    for res in group(tqdm(list(generate_fs_list())), 500):
        for f in Parallel(10, verbose=10)(delayed(upload)(f, absf, tgtf)
                                          for f, absf, tgtf in res):
            w.write(f'{f}\n')
