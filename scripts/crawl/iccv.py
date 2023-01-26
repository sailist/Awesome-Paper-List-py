from bs4 import BeautifulSoup
import requests
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


class ICCV(Crawl):
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
                    elif ctt in {'arXiv', 'video', 'dataset'}:
                        rurl = r.attrs['href']
                    elif ctt in {'bibtex'}:
                        continue
                    else:
                        print('ignore', ctt, f'of {r}')
                        pass
                    attrs[ctt] = rurl
                    if ctt == 'pdf':
                        self.append_download_item(self.year, title, rurl)

                self.append_item(self.year, title, attrs=attrs)
                rres = ' '.join(rres)


ICCV('2021', [
    'https://openaccess.thecvf.com/ICCV2021?day=2021-10-12',
    'https://openaccess.thecvf.com/ICCV2021?day=2021-10-13',
]).start()

ICCV('2019', [
    'https://openaccess.thecvf.com/ICCV2019.py?day=2019-10-29',
    'https://openaccess.thecvf.com/ICCV2019.py?day=2019-10-30',
    'https://openaccess.thecvf.com/ICCV2019.py?day=2019-10-31',
    'https://openaccess.thecvf.com/ICCV2019.py?day=2019-11-01',
]).start()

ICCV('2017', [
    'https://openaccess.thecvf.com/ICCV2017',
]).start()
