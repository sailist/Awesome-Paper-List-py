import requests
import os
from collections import Counter
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin


class Crawl:
    def __init__(self, link=None, year=None, type='strong') -> None:
        self.dic = {}
        self.link = link
        self.year = year
        self.ttype = type

    def parse(self):
        year = self.year
        res = requests.get(self.link)
        soup = BeautifulSoup(res.content.decode(), features='lxml')

        page = soup.find('section', class_='page__content')
        page.find_all('h2')

        group = [i for i in list(page.children) if i.name in {'h2', 'h3', 'p'}]

        res = []

        for item in group:
            if item.name == 'p':
                title = item.find(self.ttype)
                if title is None:
                    continue
                title = title.get_text()
                self.append_item(year, title, type=f'{first}_{sec}')
            elif item.name == 'h2':
                first = item.attrs['id'].replace('-', '_')
                sec = ''
            elif item.name == 'h3':
                sec = item.attrs['id'].replace('-', '_')

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


class Crawl2019(Crawl):
    def parse(self):
        year = self.year
        res = requests.get(self.link)
        soup = BeautifulSoup(res.content.decode())
        lis = soup.find_all('p', class_='align-items-stretch')
        root = 'https://aclanthology.org/'
        for item in lis:
            title = item.strong.get_text()
            link = item.strong.a.attrs['href']
            attrs = {'link': urljoin(root, link)}

            for k in lis[0].find('span',
                                 class_='list-button-row').find_all('a'):
                link = k.attrs['href']
                if 'http' not in link:
                    link = urljoin(root, link)
                key = k.get_text()
                attrs[key] = link
            self.append_item(year, title, attrs=attrs)


class Crawl2022(Crawl):
    def parse(self):
        year = '2022'
        fs = os.listdir(path='.')
        fs = [i for i in fs if i.endswith('html')]
        for name in fs:
            k = os.path.splitext(name)[0].split('_', maxsplit=1)
            with open(name) as r:
                soup = BeautifulSoup(r.read())
            ps = soup.find_all('td', class_='footable-first-visible')
            for item in ps:
                title = item.get_text().strip()
                self.append_item(year, title=title, type=k[1])


# Crawl('https://acl2020.org/program/accepted/', '2020', 'b').start()
# Crawl('https://2021.aclweb.org/program/accept/', '2021', 'strong').start()
# Crawl2019('https://aclanthology.org/events/acl-2019/', '2019').start()
Crawl2022().start()