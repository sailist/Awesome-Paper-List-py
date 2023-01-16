import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
from urllib.parse import urljoin
from collections import Counter


class Crawl:
    def __init__(self) -> None:
        self.dic = {}

    def parse(self):
        root = "https://papers.nips.cc/"
        res = requests.get(root)
        soup = BeautifulSoup(res.content.decode(), features="lxml")
        lists = soup.find('div', class_='col-sm').find_all('a')
        match_year = re.compile('([0-9]{4})')
        for item in tqdm(lists):
            link = urljoin(root, item.attrs['href'])
            year = match_year.search(item.get_text()).group()
            res = requests.get(link)
            content = res.content.decode()
            soup = BeautifulSoup(content, features="lxml")
            ps = soup.find('div', class_='container-fluid').find_all('li')

            for item in ps:
                item = item.a
                link = item.attrs["href"]
                self.append_item(year,
                                 item.get_text(),
                                 attrs={'link': link})

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