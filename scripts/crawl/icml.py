import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from collections import Counter
from base import Crawl


class ICML(Crawl):
    def __init__(self, year, link) -> None:
        super().__init__()
        self.year = year
        self.link = link

    def get_soup(self):
        res = requests.get(self.link)
        return BeautifulSoup(res.content.decode(), features="lxml")

    def parse2022(self):
        soup = self.get_soup()
        ps = soup.find_all('div', class_='maincard')
        links = [[
            i.find('div', class_='maincardBody').get_text().strip(),
            i.find('a', class_='href_PDF')
        ] for i in ps]

        for item in links:
            if item[1] is None:
                continue
            self.append_item('2022', item[0], {'link': item[1].attrs['href']})

    def parse2021(self):
        soup = self.get_soup()
        ps = soup.find_all('div', class_='maincard')
        links = [[
            i.find('div', class_='maincardBody').get_text().strip(),
            i.find('a', class_='paper-pdf-link')
        ] for i in ps]

        for item in links:
            if item[1] is None:
                continue
            self.append_item('2021', item[0], {'link': item[1].attrs['href']})

    def parse2020(self):
        soup = self.get_soup()
        ps = soup.find_all('div', class_='maincard')
        links = [[
            i.find('div', class_='maincardBody').get_text().strip(),
        ] for i in ps]

        for item in links:
            self.append_item('2020', item[0])

    def parse(self):
        res = requests.get(self.link)
        soup = BeautifulSoup(res.content.decode(), features='lxml')
        ps = soup.find_all('div', class_='paper')
        for item in ps:
            title = item.find('p', class_='title').get_text().strip()
            attrs = {}
            for a in item.find_all('a'):
                at = a.get_text().strip()
                link = a.attrs['href']
                if 'http' not in link:
                    link = urljoin(self.link, link)

                if at == 'abs':
                    attrs['link'] = link
                elif at == 'Download PDF':
                    attrs['pdf'] = link
                    self.append_download_item(
                        self.year, title, link)
                elif 'Supplementary' in at:
                    attrs['supp'] = link
                elif at == 'Code' or at == 'Code for experiments':
                    attrs['code'] = link
                elif at in {'Other Files','Software'}:
                    attrs[at] = link
                else:
                    print(f'ignore {a}')
            self.append_item(self.year, title, attrs)


ICML("2022", "https://proceedings.mlr.press/v162/").start()
ICML("2021", "https://proceedings.mlr.press/v139/").start()
ICML("2020", "https://proceedings.mlr.press/v119/").start()
ICML("2019", "https://proceedings.mlr.press/v97/").start()
ICML("2018", "https://proceedings.mlr.press/v80/").start()
ICML("2017", "https://proceedings.mlr.press/v70/").start()
ICML("2016", "https://proceedings.mlr.press/v48/").start()
ICML("2015", "https://proceedings.mlr.press/v37/").start()
ICML("2014", "https://proceedings.mlr.press/v32/").start()
ICML("2013", "https://proceedings.mlr.press/v28/").start()
ICML("2011", "https://proceedings.mlr.press/v27/").start()
