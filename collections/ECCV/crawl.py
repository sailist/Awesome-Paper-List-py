import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
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
    def __init__(self) -> None:
        self.dic = {}

    def parse(self):
        res = requests.get("https://www.ecva.net/papers.php")
        soup = BeautifulSoup(res.content.decode())
        years = [
            list(i.dl.children)
            for i in soup.find_all('div', class_='accordion-content')
        ]
        root = 'https://www.ecva.net/'
        for year_papers in years:
            paper_group = list(iter_paper(year_papers))
            for g in paper_group:
                if len(g) == 0 or len(g) == 1 or g[0].name != 'dt':
                    continue

                href = g[0].a

                title = href.get_text().strip()
                url = urljoin(root, href.attrs['href'])

                refs = g[2].find_all('a')

                attrs = {'link': url}
                for r in refs:
                    ctt = r.get_text()
                    if ctt in {'pdf', 'supplementary material'}:
                        rurl = urljoin(root, r.attrs['href'])
                    elif ctt in {'arXiv', 'video'}:
                        rurl = r.attrs['href']
                    elif ctt in {'bibtex', 'DOI'}:
                        continue
                    else:
                        print('ignore', ctt, f'of {r}')
                        continue

                    attrs[ctt] = rurl

                year = re.search('eccv_([0-9]{4})', url).group(1)
                self.append_item(year, title, attrs)


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