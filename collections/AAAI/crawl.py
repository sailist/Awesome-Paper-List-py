import fitz  # pip install pymupdf
import os
from bs4 import BeautifulSoup
import re
from collections import Counter


def iter_group(div):
    res = []
    for i in list(div.children):
        if i == '\n':
            continue

        rstr = i.get_text()

        if rstr is not None:
            if re.search(' *[0-9]+:.*', rstr):
                yield ' '.join(res)
                res = []
            elif re.search('[,;()]', rstr):
                continue
            rstr = re.sub('-­‐', '-', rstr)
            res.append(' '.join(rstr.split()))
    yield ' '.join(res)


class Crawl:
    def __init__(self, year, file) -> None:
        self.dic = {}
        self.year = year
        self.file = file

    def parse(self):

        doc = fitz.open(self.file)
        res = []
        for page in doc:
            soup = BeautifulSoup(page.get_text('html'))
            res.extend(list(iter_group(soup.div)))

        res = [i.strip() for i in res]
        res = [i for i in res if re.search('^[0-9]+:', i)]
        res = [re.sub('^[0-9]+: +', '', i) for i in res]
        for item in res:
            self.append_item(self.year, item)

    def start(self):
        self.parse()
        self.write()

    def append_item(self, year, title, attrs=None, type=None):
        if type is not None:
            self.dic.setdefault((year, type), []).append([title, attrs])
        else:
            self.dic.setdefault((year, ''), []).append([title, attrs])

    def write(self):
        cc = Counter()
        for k, v in self.dic.items():
            year = k[0]
            k = '_'.join([i for i in k if i is not None]).strip('_')
            res = []
            for i, (title, attrs) in enumerate(v, start=1):
                if attrs is None:
                    res.append(f'{i}. {title}')
                else:
                    attrs = ', '.join(
                        [f"[{k}]({v})" for k, v in attrs.items()])
                    res.append(f'{i}. {title} | {attrs}')
            cc[year] += len(res)
            with open(f'{k}.md', 'w') as w:
                w.write('\n'.join(res))
                print(f' - write {len(res)} papers for {k}.')
        for k, v in cc.items():
            print(f'total {v} papers in {k}.')


Crawl('2019', '../../archive/AAAI-19_Accepted_Papers.pdf').start()

Crawl('2020', '../../archive/AAAI-20-Accepted-Paper-List.pdf').start()

Crawl('2021',
      '../../archive/AAAI-21_Accepted-Paper-List.Main_.Technical.Track_.pdf').start()

Crawl('2022',
      '../../archive/AAAI-22_Accepted_Paper_List_Main_Technical_Track.pdf').start()
