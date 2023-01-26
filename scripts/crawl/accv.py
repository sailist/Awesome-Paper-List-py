import re
import os
from joblib import Parallel, delayed
from bs4 import BeautifulSoup
import requests
from collections import Counter
from urllib.parse import urljoin
from base import Crawl

def iter_paper(c):
    tmp = []
    for i in c:
        if i.name == 'dt':
            yield tmp
            tmp = []
        if i != '\n':
            tmp.append(i)
    yield tmp


class ACCV(Crawl):
    def __init__(self, year, links) -> None:
        super().__init__()
        self.year = year
        self.links = links
    
    def parse(self):
        root = 'https://openaccess.thecvf.com/'
        for link in self.links:
            res = requests.get(link)
            content = res.content.decode()
            soup = BeautifulSoup(content, features="lxml")
            paper_group = list(iter_paper(soup.dl.children))
            for g in paper_group:
                if len(g) == 0 or len(g) == 1 or g[0].name != 'dt':
                    continue

                href = g[0].a

                title = href.get_text().strip()
                url = urljoin('https://openaccess.thecvf.com/',
                              href.attrs['href'])

                attrs = {'link': url}
                refs = g[2].find_all('a')
                rres = [f'[[link]({url})]']
                for r in refs:
                    ctt = r.get_text()
                    if ctt in {'pdf', 'supp'}:
                        rurl = urljoin(root, r.attrs['href'])
                    elif ctt in {'arXiv', 'video', 'dataset','code'}:
                        rurl = r.attrs['href']
                    elif ctt in {'bibtex'}:
                        continue
                    else:
                        print('ignore', ctt, f'of {r}')
                        pass
                    
                    if ctt == 'pdf':
                        self.append_download_item(self.year,title,rurl)
                    attrs[ctt] = rurl
                self.append_item(self.year, title, attrs=attrs)
                rres = ' '.join(rres)

    
        
ACCV('2022', [
    'https://openaccess.thecvf.com/ACCV2022',
]).start()

ACCV('2020', [
    'https://openaccess.thecvf.com/ACCV2020',
]).start()


