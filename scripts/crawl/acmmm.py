import re
from bs4 import BeautifulSoup
import requests
from collections import Counter
from base import Crawl, COLLECTION_ROOT


class ACMMM(Crawl):
    def parse_2020(self):
        res = requests.get("https://2020.acmmm.org/main-track-list.html")
        soup = BeautifulSoup(res.content.decode(), features="lxml")
        ps = soup.find_all('p')
        ps = [p.get_text().strip() for p in ps]
        ps = [re.sub('^([0-9]+) +- +', '', i) for i in ps]
        for item in ps:
            self.append_item('2020', item)

    def parse_2021(self):
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

    def parse(self):
        self.parse_2020()
        self.parse_2021()


ACMMM().start()
