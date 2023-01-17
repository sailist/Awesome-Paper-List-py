from bypy import ByPy
from tqdm import tqdm
import os


def preprocess(f,root):
    """预处理
     - 字母小写
     - 删除两侧空格
     - 出现同名则删除该论文
    """
    bs, ext = os.path.splitext(f)
    bs = bs.lower().strip()
    af = f'{bs}{ext}'
    if f != af:
        if os.path.exists(os.path.join(root,af)):
            os.remove(os.path.join(root,f))
        else:    
            os.rename(os.path.join(root,f), os.path.join(root,af))
    return af

# 只在当前目录下同步 pdf 文件
cur_dir = os.path.dirname(__file__)
sync_f = os.path.join(cur_dir,'.sync')


fs = os.listdir(cur_dir)
fs = set(preprocess(i,cur_dir) for i in fs if i.endswith('pdf'))

# 同步到百度云
bp = ByPy()

if os.path.exists(sync_f):
    with open(sync_f) as r:
        sync_fs = set([i.strip() for i in r.readlines()])
        fs = fs - sync_fs # 忽略百度云删除的情况，全量同步删除 .sync 即可
    


if len(fs) > 0:
    for f in tqdm(fs):
        src = os.path.join(cur_dir,f)
        tgt = os.path.join('paper-pdfs',f)
        bp.upload(src, tgt)
        

    with open(sync_f,'a') as w:
        w.write(''.join([f'{f}\n' for f in fs]))