from collections import Counter
from datetime import datetime
import pandas as pd
import os
import re
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(__file__))
COLLECTION_ROOT = os.path.join(ROOT, 'collections')
CUR_DIR = os.path.dirname(__file__)

# edit this, then execute.
match_lis = [
    re.compile('emotion [a-z]+|sentiment [a-z]+'),
    re.compile('pretrain'),
    re.compile('contrastive'),
]

rows = []

process = re.compile('^.+\. +([^|]*)(\|.*)?')

patterns = ' & '.join([i.pattern for i in match_lis])

for root, dirs, fs in tqdm(os.walk(COLLECTION_ROOT)):
    fs = [i for i in fs if i.endswith('md')]
    for f in fs:
        absf = os.path.join(root, f)
        try:
            year = re.search('([0-9]{4})', f).group()
        except:
            print(f'Ignore {absf}')
            continue
        conf = os.path.basename(os.path.dirname(absf))
        with open(absf, encoding='utf-8') as r:
            for line in r.readlines():
                line = line.lower().strip().replace(':', '-')
                line = process.sub(r'\1', line)
                match_res = [i.search(line) for i in match_lis]
                if any(match_res):
                    match_f = [(j.group(), 2**score) for i, j, score in zip(

                        match_lis, match_res, reversed(range(len(match_res))))
                        if j is not None]
                    patterns = ' & '.join([i[0] for i in match_f])
                    score = sum([i[1] for i in match_f])
                    rows.append([conf, line, patterns, absf, score, year])

rows.sort(key=lambda x: (x[-2], x[-1]), reverse=True)

df = pd.DataFrame(
    rows, columns=['conf', 'title', 'pattern', 'source', 'score', 'year'])
df = df[['year', 'conf', 'score', 'title', 'pattern', 'source']]
name = datetime.now().strftime('%Y%M%d_%H%M%S.xlsx')
fpath = os.path.join(CUR_DIR, name)
df.to_excel(fpath)

print(f'Collect {len(df)} papers with distribution {Counter(df["score"])}.')

print(fpath)
