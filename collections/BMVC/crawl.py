import re
from bs4 import BeautifulSoup
import requests
from collections import Counter


class Crawl:
    def __init__(self, year, link) -> None:
        self.dic = {}
        self.year = year
        self.link = link

    def parse(self):
        res = requests.get(self.link)
        soup = BeautifulSoup(res.content.decode())
        ps = soup.find('table', class_='table')

        paper_group = ps.find_all('tr')
        for g in paper_group:
            if len(g) == 1:
                continue

            title = g.strong.get_text().strip()

            refs = g.find_all('a')

            oral = None

            attrs = {}
            for r in refs:
                ctt = r.get_text()
                if ctt in {'Paper', 'Supplemental', 'Code'}:
                    rurl = r.attrs['href']
                elif 'Poster Session' in ctt:
                    continue
                elif 'Oral Session' in ctt:
                    oral = 'Oral'
                    continue
                else:
                    print('ignore', ctt, f'of {r}')
                    continue
                attrs[ctt] = rurl

            self.append_item(self.year, title, attrs, type=oral)

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


Crawl('2021',
      "https://www.bmvc2021-virtualconference.com/programme/accepted-papers/"
      ).start()
Crawl(
    '2020',
    "https://www.bmvc2020-conference.com/programme/accepted-papers/").start()
