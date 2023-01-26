import re
import os
from joblib import Parallel, delayed
from bs4 import BeautifulSoup
import requests
from collections import Counter
from base import Crawl


class BMVC(Crawl):
    def __init__(self, year, link) -> None:
        super().__init__()
        self.year = year
        self.link = link

    def parse(self):
        res = requests.get(self.link)
        soup = BeautifulSoup(res.content.decode(), features='lxml')
        ps = soup.find('table', class_='table')

        paper_group = ps.find_all('tr')
        for g in paper_group:
            if len(g) == 1:
                continue

            title = g.find_all('td')[1].strong.get_text().strip()
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

                if ctt == 'Paper':
                    self.append_download_item(self.year, title, rurl)

            self.append_item(self.year, title, attrs, type=oral)


BMVC('2021',
     "https://www.bmvc2021-virtualconference.com/programme/accepted-papers/"
     ).start()
BMVC(
    '2020',
    "https://www.bmvc2020-conference.com/programme/accepted-papers/").start()
