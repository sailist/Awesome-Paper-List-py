import requests
from bs4 import BeautifulSoup
import re
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


class ECCV(Crawl):

    def parse(self):
        res = requests.get("https://www.ecva.net/papers.php")
        soup = BeautifulSoup(res.content.decode(), features='lxml')
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
                year = re.search('eccv_([0-9]{4})', url).group(1)
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
                    if ctt == 'pdf':
                        self.append_download_item(year, title, rurl)

                self.append_item(year, title, attrs)


ECCV().start()
