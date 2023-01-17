import re
from bs4 import BeautifulSoup
import requests
from collections import Counter


class Crawl:
    def __init__(self) -> None:
        self.dic = {}
        
    def parse(self):
        pass

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


class Crawl2020(Crawl):
    def parse(self):
        res = requests.get("https://2020.acmmm.org/main-track-list.html")
        soup = BeautifulSoup(res.content.decode(), features="lxml")
        ps = soup.find_all('p')
        ps = [p.get_text().strip() for p in ps]
        ps = [re.sub('^([0-9]+) +- +', '', i) for i in ps]
        for item in ps:
            self.append_item('2020', item)


class Crawl2021(Crawl):
    def parse(self):
        with open('2021.html') as r:
            soup = BeautifulSoup(r.read(), features="lxml")
            ps = soup.find_all('p')
            with open('2021.md', 'w') as w:
                ps = [p.get_text().strip() for p in ps]
                ps = [
                    re.sub('^([0-9]+) *', '', i) for i in ps
                    if re.search('^([0-9]+) *', i)
                ]
                for item in ps:
                    self.append_item('2021', item)


Crawl2021().start()
Crawl2020().start()