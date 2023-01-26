import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from collections import Counter
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

class WACV(Crawl):
    def __init__(self, year, links) -> None:
        super().__init__()
        self.year = year
        self.links = links

    def parse(self):
        for link in self.links:
            res = requests.get(link)
            content = res.content.decode()
            soup = BeautifulSoup(content, features="lxml")
            paper_group = list(iter_paper(soup.dl.children))

            for g in paper_group:
                root = 'https://openaccess.thecvf.com/'

                if len(g) == 0 or len(g) == 1 or g[0].name != 'dt':
                    continue

                href = g[0].a

                title = href.get_text().strip()
                url = urljoin('https://openaccess.thecvf.com/',
                              href.attrs['href'])

                attrs = {'link': url}

                refs = g[2].find_all('a')
                for r in refs:
                    ctt = r.get_text()
                    if ctt in {'pdf', 'supp'}:
                        rurl = urljoin(root, r.attrs['href'])
                    elif ctt in {'arXiv', 'video'}:
                        rurl = r.attrs['href']
                    elif ctt in {'bibtex'}:
                        continue
                    else:
                        print('ignore', ctt, f'of {r}')
                        pass
                    
                    attrs[ctt] = rurl
                    if ctt == 'pdf':
                        self.append_download_item(self.year, title, rurl)

                self.append_item(self.year, title, attrs)



WACV('2023', [
    'https://openaccess.thecvf.com/WACV2023',
]).start()

WACV('2022', [
    'https://openaccess.thecvf.com/WACV2022',
]).start()

WACV('2021', [
    'https://openaccess.thecvf.com/WACV2021',
]).start()

WACV('2020', [
    'https://openaccess.thecvf.com/WACV2020',
]).start()
