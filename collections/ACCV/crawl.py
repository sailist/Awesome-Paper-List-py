import re
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from collections import Counter


def iter_paper(c):
    tmp = []
    for i in c:
        if i.name == 'dt':
            yield tmp
            tmp = []
        if i != '\n':
            tmp.append(i)
    yield tmp


class Crawl:
    def __init__(self, link=None, year=None, type='strong') -> None:
        self.dic = {}
        self.link = link
        self.year = year
        self.ttype = type

    def parse(self):
        root = 'https://www.ecva.net/'
        res = requests.get('https://www.ecva.net/papers.php')
        soup = BeautifulSoup(res.content.decode())
        res = soup.find_all('dl')
        for r in res:
            year = re.search('20[0-9][0-9]', r.a.attrs['href']).group()
            print(f'Process {year}')
            paper_group = list(iter_paper(r.children))
            for g in paper_group:
                if len(g) == 0 or len(g) == 1 or g[0].name != 'dt':
                    continue

                href = g[0].a

                title = href.string
                url = urljoin(root, href.attrs['href'])

                mstr = f'{title}'.strip()
                attrs = {'link': url}
                refs = g[2].find_all('a')
                for r in refs:
                    if 'href' not in r.attrs:
                        continue
                    link = r.attrs['href']
                    if r.string in {'bibtex', 'DOI'}:
                        continue
                    if 'http' not in link:
                        link = urljoin(root, link)

                    attrs[r.get_text()] = link
                self.append_item(year, mstr, attrs=attrs)

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


Crawl().start()